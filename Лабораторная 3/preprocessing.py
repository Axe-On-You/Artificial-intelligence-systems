"""Стандартизация признаков по статистикам обучающей выборки."""

import numpy as np


class StandardScaler:
    def fit(self, x):
        values = self._prepare(x)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        # Постоянный признак после вычитания среднего равен нулю.
        self.scale_[self.scale_ == 0] = 1.0
        return self

    @staticmethod
    def _prepare(x):
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
            raise ValueError("X должен быть непустой конечной матрицей")
        return values

    def transform(self, x):
        if not hasattr(self, "mean_"):
            raise RuntimeError("Сначала вызовите fit")
        values = self._prepare(x)
        if values.shape[1] != len(self.mean_):
            raise ValueError("Число признаков отличается от обучающей выборки")
        return (values - self.mean_) / self.scale_

    def fit_transform(self, x):
        return self.fit(x).transform(x)
