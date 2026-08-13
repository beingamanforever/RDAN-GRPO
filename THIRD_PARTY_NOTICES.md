# Third-party notices

## Rubrics-To-Tokens

Repository: <https://github.com/TURLEing/Rubrics-To-Tokens>

Inspected revision: `b1ab2fba9bece98674e5fa6e6c808d9d63235778`

The response-group normalization, rubric-token normalization, and clipped actor-loss design in this project are adapted from the corresponding concepts in Rubrics-To-Tokens.
Rubrics-To-Tokens is licensed under the Apache License 2.0.
Its license is reproduced in `third_party/licenses/RTT-APACHE-2.0.txt`.

The conditional quality channel is an original implementation inspired by PAPO's published dual-advantage method.
No PAPO source code is copied, and this project does not claim to reproduce either paper.

## Alibaba ROLL

Repository: <https://github.com/alibaba/ROLL>

Inspected revision: `3077befc5f14157ab292b45809f85f2707630b91`

The compatibility functions `patch_torch_find_nd_overlapping_shards` and `patch_torch_validate_global_plan` are adapted from `mcore_adapter/src/mcore_adapter/patcher.py` at the inspected revision.
Alibaba ROLL is licensed under the Apache License 2.0.

## HIR-16K

Dataset: <https://huggingface.co/datasets/sastpg/HIR-16K>

Pinned revision: `2a95f69eb56cc47edc16a45f939cde479673a4cb`

HIR-16K is licensed under the Apache License 2.0.
The repository manifest records the released artifact hash and the byte-exact transformation to the RTT processed schema.
Dataset bytes are fetched locally and are not committed to this repository.

## AdvancedIF

Dataset: <https://huggingface.co/datasets/facebook/AdvancedIF>

Evaluator: <https://github.com/facebookresearch/AdvancedIF>

Pinned dataset revision: `e20cba9b94b59c027dfab00b29244e8bc42e4ab4`

Pinned evaluator revision: `f9d30137c4139d4d9af260ae28108b5afae828c0`

AdvancedIF is licensed under CC BY-NC 4.0.
Its bytes are fetched only for contamination checks and evaluation and are not committed to this repository.
