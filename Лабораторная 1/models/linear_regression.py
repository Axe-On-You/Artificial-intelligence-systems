import numpy as np
import torch


class LinearRegression:
    """Линейная регрессия с полным градиентным спуском."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 5000,
    ):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate должен быть положительным.")

        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs должен быть положительным целым.")

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights_ = torch.empty(0, dtype=torch.float64)
        self.bias_ = torch.tensor(0.0, dtype=torch.float64)
        self.history_: list[float] = []
        self.is_fitted_ = False

    @staticmethod
    def _prepare_x(x) -> torch.Tensor:
        values = np.asarray(x, dtype=np.float64)

        if values.ndim != 2:
            raise ValueError("x должен быть матрицей формы (N, D).")

        if values.shape[1] == 0:
            raise ValueError("В x должен быть хотя бы один признак.")

        if not np.isfinite(values).all():
            raise ValueError("x содержит NaN или inf.")

        return torch.tensor(values, dtype=torch.float64)

    def fit(self, x, y):
        self.is_fitted_ = False

        x_tensor = self._prepare_x(x)
        y_values = np.asarray(y, dtype=np.float64)

        if y_values.ndim != 1:
            raise ValueError("y должен быть вектором формы (N,).")

        if len(y_values) != len(x_tensor) or len(y_values) == 0:
            raise ValueError(
                "x и y должны содержать одинаковое ненулевое "
                "количество объектов."
            )

        if not np.isfinite(y_values).all():
            raise ValueError("y содержит NaN или inf.")

        y_tensor = torch.tensor(y_values, dtype=torch.float64)

        n_samples, n_features = x_tensor.shape

        # Каждый вызов fit начинает обучение заново.
        self.weights_ = torch.zeros(n_features, dtype=torch.float64)
        self.bias_ = torch.tensor(0.0, dtype=torch.float64)
        self.history_ = []

        for epoch in range(self.epochs):
            predictions = x_tensor @ self.weights_ + self.bias_
            errors = predictions - y_tensor

            # Градиенты MSE по формулам из презентации.
            dw = (2.0 / n_samples) * (x_tensor.T @ errors)
            db = 2.0 * errors.mean()

            # Шаг градиентного спуска.
            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            # Сохраняем loss после обновления параметров.
            updated_predictions = (
                x_tensor @ self.weights_ + self.bias_
            )
            loss = ((updated_predictions - y_tensor) ** 2).mean()

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Loss стал NaN или inf на эпохе {epoch + 1}. "
                    "Попробуй уменьшить learning_rate."
                )

            self.history_.append(loss.item())

        self.is_fitted_ = True
        return self

    def predict(self, x) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Сначала вызови fit.")

        x_tensor = self._prepare_x(x)

        if x_tensor.shape[1] != len(self.weights_):
            raise ValueError(
                "Количество признаков отличается от обучающей выборки."
            )

        predictions = x_tensor @ self.weights_ + self.bias_
        return predictions.numpy()