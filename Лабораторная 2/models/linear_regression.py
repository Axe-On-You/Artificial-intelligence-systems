import numpy as np
import torch

from .regularization import check_penalties, penalty, penalty_gradient


class LinearRegression:
    """Линейная регрессия с ручным градиентом MSE и штрафом L1 + L2."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        *,
        l1: float = 0.0,
        l2: float = 0.0,
    ):

        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate должен быть положительным.")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs должен быть положительным целым.")
        check_penalties(l1, l2)
        self.validation_loss_history_ = None
        self.data_loss_history_ = None
        self.history_ = None
        self.bias_ = None
        self.weights_ = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l1 = l1
        self.l2 = l2
        self.is_fitted_ = False

    @staticmethod
    def _prepare_x(x) -> torch.Tensor:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("x должен быть матрицей формы (N, D), D > 0.")
        if not np.isfinite(values).all():
            raise ValueError("x содержит NaN или inf.")
        return torch.tensor(values, dtype=torch.float64)

    @staticmethod
    def _prepare_y(y, n_samples: int) -> torch.Tensor:
        values = np.asarray(y, dtype=np.float64)
        if n_samples == 0 or values.shape != (n_samples,) or not np.isfinite(values).all():
            raise ValueError("y должен содержать по одному конечному числу на объект.")
        return torch.tensor(values, dtype=torch.float64)

    def fit(self, x, y, *, validation_data=None):
        self.is_fitted_ = False
        x_train = self._prepare_x(x)
        n_samples, n_features = x_train.shape
        y_train = self._prepare_y(y, n_samples)
        if validation_data is not None:
            x_val = self._prepare_x(validation_data[0])
            y_val = self._prepare_y(validation_data[1], len(x_val))
            if x_val.shape[1] != n_features:
                raise ValueError("Количество признаков train и validation должно совпадать.")

        self.weights_ = torch.zeros(n_features, dtype=torch.float64)
        self.bias_ = torch.tensor(0.0, dtype=torch.float64)
        self.history_ = []
        self.data_loss_history_ = []
        self.validation_loss_history_ = []

        for epoch in range(self.epochs):
            errors = x_train @ self.weights_ + self.bias_ - y_train
            dw = (2.0 / n_samples) * (x_train.T @ errors)
            db = 2.0 * errors.mean()
            dw += penalty_gradient(self.weights_, self.l1, self.l2)

            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            train_mse = (x_train @ self.weights_ + self.bias_ - y_train).square().mean()
            objective = train_mse + penalty(self.weights_, self.l1, self.l2)
            if not torch.isfinite(objective).item():
                raise FloatingPointError(f"Функция потерь расходится на эпохе {epoch + 1}.")

            self.history_.append(objective.item())
            self.data_loss_history_.append(train_mse.item())
            if validation_data is not None:
                val_mse = (x_val @ self.weights_ + self.bias_ - y_val).square().mean()
                self.validation_loss_history_.append(val_mse.item())

        self.is_fitted_ = True
        return self

    def predict(self, x) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Сначала вызови fit.")
        values = self._prepare_x(x)
        if values.shape[1] != len(self.weights_):
            raise ValueError("Число признаков отличается от обучающей выборки.")
        return (values @ self.weights_ + self.bias_).numpy()
