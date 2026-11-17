"""Стратифицированное тройное разбиение по индексам."""

import numpy as np


def stratified_split_indices(y, train_size=0.6, val_size=0.2, seed=6752):
    labels = np.asarray(y)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("y должен быть непустым одномерным массивом")
    if not (0 < train_size < 1 and 0 < val_size < 1 and train_size + val_size < 1):
        raise ValueError("Доли train, validation и test должны быть положительными")
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        n_train = int(len(indices) * train_size)
        n_val = int(len(indices) * val_size)
        if min(n_train, n_val, len(indices) - n_train - n_val) < 1:
            raise ValueError(f"Недостаточно объектов класса {label} для разбиения")
        parts[0].append(indices[:n_train])
        parts[1].append(indices[n_train:n_train + n_val])
        parts[2].append(indices[n_train + n_val:])
    return tuple(rng.permutation(np.concatenate(part)) for part in parts)
