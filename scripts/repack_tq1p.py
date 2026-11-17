#!/usr/bin/env python3
"""Re-pack every TQ1_0 tensor of a TTF GGUF as TQ1_P, so it runs through the TQ1_P table kernel.

  repack_tq1p.py IN.gguf OUT.gguf

The trits are unchanged. What moves:
  - TQ1_0 keeps a scale per 256-weight block; in these files it is constant along a row, so it becomes a
    row-scale vector: <w>.ttf_r for a main weight, <w>.ttf_d_r / <w>.ttf_u_r for the ternary adapter
    (the last two need the loader lines in engine/ttf-mtp-head.patch).
  - TQ1_P needs the input width to be a multiple of 160. Wider-by-padding tensors get zero columns, and
    the vectors and float adapter on the same input (ttf_c, ttf_m, ttf_d_c, ttf_d_m, ttf_fd, ttf_fd_c,
    ttf_fd_m) are padded to match (the engine pads the input with zeros). The adapter's up matrix reads
    the rank-sized vector, which the graph pads after its column scale (also in the patch).
"""
import sys, time
import numpy as np
import torch
import gguf
from ttf_head import enc_tq1p

QT = gguf.GGMLQuantizationType
src, dst = sys.argv[1:3]
reader = gguf.GGUFReader(src)
T = {t.name: t for t in reader.tensors}
SCALE_NAME = {"weight": "ttf_r", "ttf_d": "ttf_d_r", "ttf_u": "ttf_u_r"}
pad160 = lambda k: (k + 159) // 160 * 160
dev = "cuda" if torch.cuda.is_available() else "cpu"

new, extra, pad_stems = {}, {}, {}        # name -> (array, qtype); new tensors after a name; stem -> (K, K')
t0 = time.time(); n_conv = 0; n_trits = 0
for t in reader.tensors:
    stem, _, suffix = t.name.rpartition(".")
    if t.tensor_type != QT.TQ1_0 or suffix not in SCALE_NAME:
        continue
    K, N = int(t.shape[0]), int(t.shape[1])
    blocks = np.asarray(t.data).reshape(N, K // 256, 54)
    d = blocks[:, :, 52:54].copy().view(np.float16).reshape(N, K // 256).astype(np.float32)
    scale = d.max(axis=1)                             # an all-zero block is stored with scale 0
    assert np.all((d == scale[:, None]) | (d == 0)), f"{t.name}: block scale varies along a row"
    w = torch.from_numpy(gguf.quants.dequantize(np.asarray(t.data), QT.TQ1_0).reshape(N, K)).to(dev)
    trits = torch.sign(w).to(torch.int8)
    Kp = pad160(K)
    if Kp != K:
        trits = torch.cat([trits, torch.zeros(N, Kp - K, dtype=torch.int8, device=dev)], 1)
        if suffix in ("weight", "ttf_d"):
            assert pad_stems.setdefault(stem, (K, Kp)) == (K, Kp)
    new[t.name] = (enc_tq1p(trits.cpu()), QT.TQ1_P)
    sname = stem + "." + SCALE_NAME[suffix]
    if sname in T:                                   # an existing row scale: combine
        scale = scale * np.asarray(T[sname].data, dtype=np.float32)
        new[sname] = (scale.astype(np.float32), None)
    else:
        extra[t.name] = (sname, scale.astype(np.float32))
    n_conv += 1; n_trits += N * K
    if n_conv % 100 == 0:
        print(f"{n_conv} tensors, {n_trits / 1e9:.2f} B trits, {time.time() - t0:.0f}s", flush=True)

# input-side vectors and the float adapter of the padded linears
for stem, (K, Kp) in pad_stems.items():
    for suf, fill in (("ttf_c", 1.0), ("ttf_m", 0.0), ("ttf_d_c", 1.0), ("ttf_d_m", 0.0), ("ttf_fd_c", 1.0), ("ttf_fd_m", 0.0)):
        name = f"{stem}.{suf}"
        if name in T:
            v = np.asarray(T[name].data, dtype=np.float32)
            assert v.shape == (K,), (name, v.shape)
            new[name] = (np.concatenate([v, np.full(Kp - K, fill, dtype=np.float32)]), None)
    name = f"{stem}.ttf_fd"
    if name in T:
        a = np.asarray(T[name].data)                   # numpy shape [rank, K]
        assert a.shape[1] == K and a.dtype == np.float16, (name, a.shape, a.dtype)
        new[name] = (np.concatenate([a, np.zeros((a.shape[0], Kp - K), dtype=np.float16)], 1), None)
print(f"converted {n_conv} tensors ({n_trits / 1e9:.2f} B trits), padded linears {len(pad_stems)}, {time.time() - t0:.0f}s", flush=True)

arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
align = reader.get_field(gguf.Keys.General.ALIGNMENT)
if align is not None:
    writer.data_alignment = align.contents()
for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    vt = field.types[0]
    st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), vt, sub_type=st)

order = []
for t in reader.tensors:
    if t.name in new:
        arr, qt = new[t.name]
        order.append((t.name, arr, qt))
    else:
        order.append((t.name, t.data, t.tensor_type))
    if t.name in extra:
        order.append((extra[t.name][0], extra[t.name][1], None))
for name, arr, qt in order:
    if isinstance(qt, QT) and arr.dtype == np.uint8:
        writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, qt)
    elif qt is None:
        writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes)
    else:
        writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, qt)
writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
for name, arr, qt in order:
    writer.write_tensor_data(arr)
writer.close()
left = sum(1 for n, a, q in order if q == QT.TQ1_0)
print(f"wrote {dst}: {len(order)} tensors, TQ1_0 tensors left: {left}")
