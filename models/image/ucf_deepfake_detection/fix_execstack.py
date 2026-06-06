import os
so_path = '/usr/local/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so'
with open(so_path, 'rb') as f:
    data = bytearray(f.read())

e_phoff = int.from_bytes(data[0x20:0x28], 'little')
e_phentsize = int.from_bytes(data[0x36:0x38], 'little')
e_phnum = int.from_bytes(data[0x38:0x3a], 'little')

for i in range(e_phnum):
    offset = e_phoff + i * e_phentsize
    p_type = int.from_bytes(data[offset:offset+4], 'little')
    if p_type == 0x6474e551:
        p_flags = int.from_bytes(data[offset+4:offset+8], 'little')
        p_flags &= ~1
        data[offset+4:offset+8] = p_flags.to_bytes(4, 'little')
        break

with open(so_path, 'wb') as f:
    f.write(data)
print('Fixed PT_GNU_STACK for libtorch_cpu.so')
