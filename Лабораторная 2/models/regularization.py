import math

import torch


def check_penalties(l1: float, l2: float) -> None:
    if not all(math.isfinite(value) and value >= 0 for value in (l1, l2)):
        raise ValueError("Коэффициенты регуляризации должны быть конечными и неотрицательными.")


def penalty(weights: torch.Tensor, l1: float, l2: float) -> torch.Tensor:
    """Штраф L1 + L2 для весов, без свободного члена."""
    return l1 * weights.abs().sum() + l2 * weights.square().sum()


def penalty_gradient(weights: torch.Tensor, l1: float, l2: float) -> torch.Tensor:
    """В нуле выбираем субградиент L1, равный нулю."""
    return l1 * weights.sign() + 2 * l2 * weights
