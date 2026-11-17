"""Линейный soft-margin SVM: тензоры PyTorch и ручной субградиент."""

import math

import numpy as np
import torch


class LinearSVM:
    """Минимизирует l2/2 * ||w||² + mean(max(0, 1 - y * score))."""

    def __init__(self, learning_rate=0.01, epochs=500, batch_size=32,
                 l2=0.01, seed=6752):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate должен быть положительным и конечным")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs должен быть положительным целым")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size должен быть положительным целым")
        if not math.isfinite(l2) or l2 < 0:
            raise ValueError("l2 должен быть неотрицательным и конечным")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed
        self.is_fitted_ = False

    @staticmethod
    def _prepare_x(x):
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
            raise ValueError("X должен быть непустой конечной матрицей")
        return torch.as_tensor(values, dtype=torch.float64)

    @staticmethod
    def _prepare_y(y, n):
        values = np.asarray(y, dtype=np.float64)
        if values.shape != (n,) or not np.isin(values, [-1, 1]).all():
            raise ValueError("y должен содержать по одной метке -1 или +1 на объект")
        return torch.as_tensor(values, dtype=torch.float64)

    def _losses(self, x, y):
        hinge = torch.clamp(1 - y * (x @ self.weights_ + self.bias_), min=0).mean()
        total = hinge + self.l2 * self.weights_.square().sum() / 2
        return hinge, total

    def _batch_gradient(self, x, y):
        # В точке y*score = 1 выбирается допустимый субградиент 0.
        active = (y * (x @ self.weights_ + self.bias_) < 1).to(x.dtype)
        weighted = active * y
        dw = self.l2 * self.weights_ - x.T @ weighted / len(y)
        db = -weighted.mean()
        return dw, db

    def fit(self, x, y, *, validation_data=None):
        self.is_fitted_ = False
        x_train = self._prepare_x(x)
        y_train = self._prepare_y(y, len(x_train))
        if validation_data is not None:
            x_val = self._prepare_x(validation_data[0])
            y_val = self._prepare_y(validation_data[1], len(x_val))
            if x_val.shape[1] != x_train.shape[1]:
                raise ValueError("Количество признаков train и validation не совпадает")

        self.weights_ = torch.zeros(x_train.shape[1], dtype=torch.float64)
        self.bias_ = torch.tensor(0.0, dtype=torch.float64)
        self.history_, self.hinge_history_ = [], []
        self.validation_history_, self.validation_hinge_history_ = [], []
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        for epoch in range(self.epochs):
            order = torch.randperm(len(x_train), generator=generator)
            for start in range(0, len(order), self.batch_size):
                index = order[start:start + self.batch_size]
                dw, db = self._batch_gradient(x_train[index], y_train[index])
                self.weights_ -= self.learning_rate * dw
                self.bias_ -= self.learning_rate * db
            hinge, total = self._losses(x_train, y_train)
            if not torch.isfinite(total).item():
                raise FloatingPointError(f"Функция потерь расходится на эпохе {epoch + 1}")
            self.hinge_history_.append(hinge.item())
            self.history_.append(total.item())
            if validation_data is not None:
                val_hinge, val_total = self._losses(x_val, y_val)
                self.validation_hinge_history_.append(val_hinge.item())
                self.validation_history_.append(val_total.item())
        self.is_fitted_ = True
        return self

    def decision_function(self, x):
        if not self.is_fitted_:
            raise RuntimeError("Сначала вызовите fit")
        values = self._prepare_x(x)
        if values.shape[1] != len(self.weights_):
            raise ValueError("Число признаков отличается от обучающей выборки")
        return (values @ self.weights_ + self.bias_).numpy()

    def predict(self, x):
        return np.where(self.decision_function(x) >= 0, 1, -1)
