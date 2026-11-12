import math

import numpy as np


def random_split_indices(
    n_samples: int, *, labels=None, seed: int = 5267
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Разбиение 60/20/20. Если переданы labels, делим внутри каждого класса."""
    if n_samples < 3:
        raise ValueError("Для трёх выборок нужно хотя бы три объекта.")
    rng = np.random.default_rng(seed)
    if labels is None:
        groups = [np.arange(n_samples)]
    else:
        labels = np.asarray(labels)
        if labels.shape != (n_samples,):
            raise ValueError("Нужна одна метка на объект.")
        groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]

    train_parts = []
    validation_parts = []
    test_parts = []
    for indices in groups:
        indices = rng.permutation(indices)
        n_test = math.ceil(len(indices) / 5)
        n_validation = math.ceil((len(indices) - n_test) / 4)
        boundary = n_test + n_validation
        if boundary >= len(indices):
            raise ValueError("В группе недостаточно объектов для трёх выборок.")
        test_parts.append(indices[:n_test])
        validation_parts.append(indices[n_test:boundary])
        train_parts.append(indices[boundary:])

    train = rng.permutation(np.concatenate(train_parts))
    validation = rng.permutation(np.concatenate(validation_parts))
    test = rng.permutation(np.concatenate(test_parts))
    return train, validation, test


def chronological_day_split_indices(
    dates,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Делит отсортированные уникальные даты в пропорции 60/20/20."""
    days = np.asarray(dates, dtype="datetime64[D]")
    if days.ndim != 1 or np.isnat(days).any():
        raise ValueError("Нужен одномерный набор дат без пропусков.")
    unique_days = np.unique(days)
    if len(unique_days) < 5:
        raise ValueError("Для разбиения нужно хотя бы пять разных дат.")

    validation_start = unique_days[3 * len(unique_days) // 5]
    test_start = unique_days[4 * len(unique_days) // 5]
    train = np.flatnonzero(days < validation_start)
    validation = np.flatnonzero((days >= validation_start) & (days < test_start))
    test = np.flatnonzero(days >= test_start)
    return train, validation, test
