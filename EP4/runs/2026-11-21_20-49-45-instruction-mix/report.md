# Instruction mix of `trace_photons` (static: instructions in the code, not how often they run)

| Unit | star_gpu64 | star_gpu64_sections | star_gpu32 | star_gpu32_sections |
|---|---|---|---|---|
| FP64 units | 278 | 278 | 4 | 4 |
| FP32 cores | 73 | 73 | 247 | 247 |
| special function unit (MUFU) | 13 | 13 | 12 | 12 |
| conversions | 25 | 25 | 25 | 25 |
| integer units | 660 | 671 | 624 | 677 |
| atomics (L2) | 2 | 2 | 2 | 2 |
| constant loads | 46 | 49 | 49 | 47 |
| global memory | 14 | 14 | 5 | 5 |
| local memory (stack, spills) | 5 | 5 | 20 | 20 |
| warp-wide | 6 | 6 | 6 | 6 |
| moves, special registers | 250 | 271 | 154 | 177 |
| control flow | 1261 | 1258 | 296 | 297 |
| uniform datapath | 66 | 50 | 28 | 75 |
| other | 45 | 53 | 48 | 62 |
| **total** | 2744 | 2768 | 1520 | 1656 |
| operands read from the Scene (`c[0x3]`) | 16 | 18 | 18 | 18 |

**star_gpu64, most frequent:** NOP 1063, MOV 242, IMAD 196, LOP3 184, DFMA 152, IADD 145, BRA 90, ISETP 82, DMUL 68, FSEL 54, BSSY 44, BSYNC 44

**star_gpu64_sections, most frequent:** NOP 1060, MOV 259, IMAD 195, LOP3 184, IADD 157, DFMA 152, BRA 90, ISETP 82, DMUL 68, FSEL 54, BSSY 44, BSYNC 44

**star_gpu32, most frequent:** IMAD 194, LOP3 189, MOV 147, IADD 107, BRA 102, FFMA 97, NOP 87, ISETP 81, FMUL 55, BSSY 46, BSYNC 46, FSETP 35

**star_gpu32_sections, most frequent:** IMAD 194, LOP3 189, MOV 168, IADD 160, BRA 102, FFMA 97, NOP 88, ISETP 81, FMUL 55, UIADD3 53, BSSY 46, BSYNC 46
