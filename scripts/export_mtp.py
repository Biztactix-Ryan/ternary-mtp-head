#!/usr/bin/env python3
"""Write an MTP GGUF from the stock file plus (optionally) a trained checkpoint (runbook section 10).

  export_mtp.py STOCK.gguf OUT.gguf [--ckpt best.pt]

Metadata and token_embd / output / output_norm are copied raw. The blk.64.* matrices are written
as F16 and the norm vectors as F32, from the checkpoint or, without --ckpt, from the dequantised
stock weights (the round-trip check).
"""
import argparse
import numpy as np
import torch
import gguf
from mtp_head import BLK, MATS, NORMS, read_gguf_tensors

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--ckpt")
a = ap.parse_args()

reader = gguf.GGUFReader(a.src)
names = [BLK + k + ".weight" for k in list(MATS) + list(NORMS)]
if a.ckpt:
    sd = torch.load(a.ckpt, map_location="cpu")["model"]
    new = {BLK + k + ".weight": sd["mats." + k.replace(".", "_")].float().numpy() for k in MATS}
    new.update({BLK + k + ".weight": sd["norms." + k.replace(".", "_")].float().numpy() for k in NORMS})
else:
    new = read_gguf_tensors(a.src, set(names))
for k in MATS:
    new[BLK + k + ".weight"] = new[BLK + k + ".weight"].astype(np.float16)
for k in NORMS:
    new[BLK + k + ".weight"] = new[BLK + k + ".weight"].astype(np.float32)

arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
writer = gguf.GGUFWriter(a.dst, arch=arch, endianess=reader.endianess)
align = reader.get_field(gguf.Keys.General.ALIGNMENT)
if align is not None:
    writer.data_alignment = align.contents()
for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    vt = field.types[0]
    st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), vt, sub_type=st)

for t in reader.tensors:
    if t.name in new:
        arr = new[t.name]
        assert tuple(arr.shape) == tuple(int(x) for x in reversed(t.shape)), (t.name, arr.shape, t.shape)
        writer.add_tensor_info(t.name, arr.shape, arr.dtype, arr.nbytes)
    else:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
for t in reader.tensors:
    writer.write_tensor_data(new[t.name] if t.name in new else t.data)
    print(f"{t.name:<40} {'new ' + str(new[t.name].dtype) if t.name in new else 'copied ' + t.tensor_type.name}")
writer.close()
print("wrote", a.dst)
