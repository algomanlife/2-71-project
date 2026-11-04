# Attribution and Provenance

## 2^71 project identity

**2^71 project** is an independent, rebranded continuation of the **CUDACyclone**
project by GitHub user **Dookoo2** (<https://github.com/Dookoo2/CUDACyclone>).

- Original project: CUDACyclone — "Satoshi's quest solver for CUDA" by Dookoo2
- This project: 2^71 project — technical identifier `project_2_71`, executable `project-2-71`

The rebranding renamed project-facing identifiers, file names, build targets, and
documentation. No computational algorithms (secp256k1 math, scalar multiplication,
elliptic curve operations, batch inversion, CUDA kernel logic, hashing, or search
logic) were changed.

## Upstream sources incorporated (per the original README)

| Upstream project | Author | License | Role |
|---|---|---|---|
| [VanitySearch](https://github.com/JeanLucPons/VanitySearch) | JeanLucPons | GPL-3.0 | secp256k1 math foundation |
| [VanitySearch-Bitcrack](https://github.com/FixedPaul) | FixedPaul | AGPL-3.0 | secp256k1 math / CUDA optimizations |

The original README states: "Secp256k1 math is based on the excellent work from
JeanLucPons/VanitySearch, and FixedPaul/VanitySearch-Bitcrack with major
CUDA-specific modifications." Special thanks to Jean-Luc Pons for his foundational
contributions to the cryptographic community.

## Licensing status of the original CUDACyclone repository

The original repository at github.com/Dookoo2/CUDACyclone contains **no LICENSE
file** and GitHub reports no license for it. Absent explicit licensing terms, the
code is under exclusive copyright of the original author (Dookoo2) and
redistribution requires the author's authorization.

**Licensing obligations for this repository:**

1. Code derived from VanitySearch (GPL-3.0) and/or VanitySearch-Bitcrack
   (AGPL-3.0) carries the corresponding copyleft obligations and attribution
   requirements of those licenses.
2. The donation address `bc1qtq4y9l9ajeyxq05ynq09z8p52xdmk4hqky9c8n` in the
   README belongs to the original author of CUDACyclone and is preserved for
   attribution purposes only.

Before any public distribution of this repository, a license decision compatible
with the upstream GPL-3.0 / AGPL-3.0 obligations must be made and documented.
