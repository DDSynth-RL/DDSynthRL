from __future__ import annotations

from typing import List

FIRST_OPERATOR_INDEX = 24
OPERATOR_STRIDE = 22


def dexed_numerical_param_indices() -> List[int]:
    base = [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    indexes = list(base)
    for op in range(6):
        start = FIRST_OPERATOR_INDEX + op * OPERATOR_STRIDE
        indexes.extend([start + offset for offset in range(0, 8)])
        indexes.extend([start + offset for offset in range(8, 12) if offset != 9])
        indexes.append(start + 8)
        indexes.append(start + 10)
        indexes.append(start + 11)
        indexes.append(start + 12)
        indexes.append(start + 13)
        indexes.append(start + 14)
        indexes.append(start + 15)
        indexes.append(start + 18)
        indexes.append(start + 19)
        indexes.append(start + 20)
    return sorted(set(indexes))
