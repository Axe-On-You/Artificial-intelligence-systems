import numpy as np
import torch

from .regularization import check_penalties, penalty, penalty_gradient


class LogisticRegression:
    """Бинарная логистическая регрессия с ручным градиентом BCE."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 1000,
        threshold: float = 0.5,
        *,
        l1: float = 0.0,
        l2: float = 0.0,
    ):
        self.validation_loss_history_ = None
        self.data_loss_history_ = None
        self.history_ = None
        self.bias_ = None
        self.weights_ = None
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate должен быть положительным.")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs должен быть положительным целым.")
        if not 0 < threshold < 1:
            raise ValueError("threshold должен находиться между 0 и 1.")
        check_penalties(l1, l2)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
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
        if n_samples == 0 or values.shape != (n_samples,) or not np.isin(values, [0, 1]).all():
            raise ValueError("y должен содержать одну бинарную метку на объект.")
        return torch.tensor(values, dtype=torch.float64)

    @staticmethod
    def _bce_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = logits.clamp(min=0) - targets * logits + torch.log1p(torch.exp(-logits.abs()))
        return losses.mean()

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
            logits = x_train @ self.weights_ + self.bias_
            errors = torch.sigmoid(logits) - y_train
            dw = (x_train.T @ errors) / n_samples
            db = errors.mean()
            dw += penalty_gradient(self.weights_, self.l1, self.l2)

            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            train_bce = self._bce_from_logits(x_train @ self.weights_ + self.bias_, y_train)
            objective = train_bce + penalty(self.weights_, self.l1, self.l2)
            if not torch.isfinite(objective).item():
                raise FloatingPointError(f"Функция потерь расходится на эпохе {epoch + 1}.")

            self.history_.append(objective.item())
            self.data_loss_history_.append(train_bce.item())
            if validation_data is not None:
                val_bce = self._bce_from_logits(x_val @ self.weights_ + self.bias_, y_val)
                self.validation_loss_history_.append(val_bce.item())

        self.is_fitted_ = True
        return self

    def predict_proba(self, x) -> np.ndarray:
        """Возвращает вероятность класса 1 для каждого объекта."""
        if not self.is_fitted_:
            raise RuntimeError("Сначала вызови fit.")
        values = self._prepare_x(x)
        if values.shape[1] != len(self.weights_):
            raise ValueError("Число признаков отличается от обучающей выборки.")
        return torch.sigmoid(values @ self.weights_ + self.bias_).numpy()

    def predict(self, x) -> np.ndarray:
        return (self.predict_proba(x) >= self.threshold).astype(np.int64)
