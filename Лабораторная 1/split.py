import numpy as np


def train_test_split_indices(
    n_samples: int,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает непересекающиеся индексы train и test."""
    if n_samples < 2:
        raise ValueError("Для разбиения нужны минимум два объекта.")

    if not 0 < test_size < 1:
        raise ValueError("test_size должен находиться между 0 и 1.")

    n_test = int(np.ceil(n_samples * test_size))

    if n_test >= n_samples:
        raise ValueError("После разбиения train окажется пустым.")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return train_indices, test_indices

def stratified_train_test_split_indices(
    y,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Разделяет индексы отдельно внутри каждого класса."""
    labels = np.asarray(y)

    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("y должен быть непустым одномерным массивом.")

    if not 0 < test_size < 1:
        raise ValueError("test_size должен находиться между 0 и 1.")

    rng = np.random.default_rng(random_state)

    train_parts = []
    test_parts = []

    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        class_indices = rng.permutation(class_indices)

        n_test = int(np.ceil(len(class_indices) * test_size))

        if n_test == 0 or n_test >= len(class_indices):
            raise ValueError(
                f"Недостаточно объектов класса {label} "
                "для выбранного разбиения."
            )

        test_parts.append(class_indices[:n_test])
        train_parts.append(class_indices[n_test:])

    # Перемешиваем объединённые части, чтобы классы не шли блоками.
    train_indices = rng.permutation(np.concatenate(train_parts))
    test_indices = rng.permutation(np.concatenate(test_parts))

    return train_indices, test_indices
