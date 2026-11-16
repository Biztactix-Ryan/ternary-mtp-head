#!/usr/bin/env python3
"""Write a trunk GGUF with an in-file MTP head as block N (no separate -md file).

  build_infile.py TRUNK.gguf OUT.gguf --float-ckpt best.pt        # F16 head (reference)
  build_infile.py TRUNK.gguf OUT.gguf --ttf-ckpt  ttf.pt          # ternary head (see ttf_head.py)

Copies every trunk tensor raw, sets <arch>.block_count += 1 and <arch>.nextn_predict_layers = 1, and
appends the head's tensors as blk.<N>.*. The head then uses the trunk's own token_embd and output.weight.
"""
import argparse
import numpy as np
import torch
import gguf
from mtp_head import MATS, NORMS

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--float-ckpt"); ap.add_argument("--ttf-ckpt")
a = ap.parse_args()

reader = gguf.GGUFReader(a.src)
arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
n_block = int(reader.get_field(f"{arch}.block_count").contents())
pfx = f"blk.{n_block}."

new = []   # (name, array, quant type or None)
if a.float_ckpt:
    sd = torch.load(a.float_ckpt, map_location="cpu")["model"]
    for k in MATS:
        new.append((pfx + k + ".weight", sd["mats." + k.replace(".", "_")].float().numpy().astype(np.float16), None))
    for k in NORMS:
        new.append((pfx + k + ".weight", sd["norms." + k.replace(".", "_")].float().numpy().astype(np.float32), None))
else:
    from ttf_head import ttf_tensors
    new = ttf_tensors(torch.load(a.ttf_ckpt, map_location="cpu"), pfx)

writer = gguf.GGUFWriter(a.dst, arch=arch, endianess=reader.endianess)
align = reader.get_field(gguf.Keys.General.ALIGNMENT)
if align is not None:
    writer.data_alignment = align.contents()
for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    vt = field.types[0]
    st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    val = field.contents()
    if field.name == f"{arch}.block_count":
        val = n_block + 1
    elif vt == gguf.GGUFValueType.ARRAY and field.name.startswith(arch + ".") and len(val) == n_block:
        print(f"note: per-layer array {field.name} (len {n_block}) extended with {val[-1]!r} -> False/0 for the head block")
        val = list(val) + [type(val[-1])(0)]
    writer.add_key_value(field.name, val, vt, sub_type=st)
writer.add_uint32(f"{arch}.nextn_predict_layers", 1)

for t in reader.tensors:
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
for name, arr, qt in new:
    writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, qt)
writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
for t in reader.tensors:
    writer.write_tensor_data(t.data)
for name, arr, qt in new:
    writer.write_tensor_data(arr)
    print(f"{name:<44} {list(arr.shape)} {qt.name if qt is not None else arr.dtype}")
writer.close()
print("wrote", a.dst)
