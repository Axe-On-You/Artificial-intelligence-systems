import numpy as np
import torch


class KNN:
    """K ближайших соседей для классификации и регрессии."""

    def __init__(self, k: int = 5, task: str = "classification", distance: str = "euclidean"):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k должно быть положительным целым.")
        if task not in ("classification", "regression"):
            raise ValueError("task должен быть classification или regression.")
        if distance not in ("euclidean", "manhattan"):
            raise ValueError("distance должен быть euclidean или manhattan.")
        self.k = k
        self.task = task
        self.distance = distance
        self.x_train_ = None
        self.y_train_ = None
        self.classes_ = None


    @staticmethod
    def _prepare_x(x) -> torch.Tensor:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
            raise ValueError("x должен быть матрицей конечных чисел.")
        return torch.tensor(values, dtype=torch.float64)

    def fit(self, x, y):
        x_train = self._prepare_x(x)
        targets = np.asarray(y)
        if targets.shape != (len(x_train),) or len(x_train) < self.k:
            raise ValueError("Нужна одна метка на объект и не меньше k объектов.")
        if self.task == "regression":
            targets = targets.astype(np.float64)
            if not np.isfinite(targets).all():
                raise ValueError("Ответы регрессии должны быть конечными числами.")
        else:
            self.classes_ = np.unique(targets)
        self.x_train_ = x_train
        self.y_train_ = targets.copy()
        return self

    def _distances(self, queries: torch.Tensor) -> torch.Tensor:
        differences = queries[:, None, :] - self.x_train_[None, :, :]
        if self.distance == "euclidean":
            return differences.square().sum(dim=2).sqrt()
        return differences.abs().sum(dim=2)

    def kneighbors(self, x) -> np.ndarray:
        """Возвращает индексы k ближайших train-объектов по возрастанию расстояния."""
        if self.x_train_ is None:
            raise RuntimeError("Сначала вызови fit.")
        queries = self._prepare_x(x)
        if queries.shape[1] != self.x_train_.shape[1]:
            raise ValueError("Число признаков отличается от обучающей выборки.")

        indices = np.empty((len(queries), self.k), dtype=np.int64)
        batch_size = 32
        for start in range(0, len(queries), batch_size):
            stop = min(start + batch_size, len(queries))
            distances = self._distances(queries[start:stop])
            order = torch.argsort(distances, dim=1, stable=True)
            indices[start:stop] = order[:, :self.k].numpy()
        return indices

    def predict_from_neighbors(self, indices: np.ndarray) -> np.ndarray:
        """Агрегирует метки выбранных соседей, в том числе префикса при подборе k."""
        if self.x_train_ is None:
            raise RuntimeError("Сначала вызови fit.")
        if indices.ndim != 2 or indices.shape[1] == 0:
            raise ValueError("Нужна матрица с непустыми списками соседей.")
        neighbor_targets = self.y_train_[indices]
        if self.task == "regression":
            return neighbor_targets.mean(axis=1)

        votes = np.column_stack([
            (neighbor_targets == label).sum(axis=1) for label in self.classes_
        ])
        return self.classes_[votes.argmax(axis=1)]

    def predict(self, x) -> np.ndarray:
        return self.predict_from_neighbors(self.kneighbors(x))
