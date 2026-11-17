# Собственная модель LogisticRegression из лабораторной работы № 1.
# Скопирована для автономного запуска ЛР3; алгоритм не изменён.
import numpy as np
import torch


class LogisticRegression:
    """Бинарная логистическая регрессия с ручными градиентами."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 3000,
        threshold: float = 0.5,
    ):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate должен быть положительным.")

        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs должен быть положительным целым.")

        if not 0 < threshold < 1:
            raise ValueError("threshold должен находиться между 0 и 1.")

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold

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

    @staticmethod
    def _bce_from_logits(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Устойчивая форма BCE:
        # max(z, 0) - y*z + log(1 + exp(-abs(z))).
        losses = (
            torch.clamp(logits, min=0)
            - targets * logits
            + torch.log1p(torch.exp(-torch.abs(logits)))
        )
        return losses.mean()

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

        if not np.isin(y_values, [0.0, 1.0]).all():
            raise ValueError("y должен содержать только классы 0 и 1.")

        y_tensor = torch.tensor(y_values, dtype=torch.float64)

        n_samples, n_features = x_tensor.shape

        self.weights_ = torch.zeros(n_features, dtype=torch.float64)
        self.bias_ = torch.tensor(0.0, dtype=torch.float64)
        self.history_ = []

        for epoch in range(self.epochs):
            logits = x_tensor @ self.weights_ + self.bias_
            probabilities = torch.sigmoid(logits)

            # Градиенты BCE по формулам из презентации.
            errors = probabilities - y_tensor
            dw = (x_tensor.T @ errors) / n_samples
            db = errors.mean()

            # Ручное обновление параметров.
            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            updated_logits = (
                x_tensor @ self.weights_ + self.bias_
            )
            loss = self._bce_from_logits(
                updated_logits,
                y_tensor,
            )

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Loss стал NaN или inf на эпохе {epoch + 1}. "
                    "Попробуй уменьшить learning_rate."
                )

            self.history_.append(loss.item())

        self.is_fitted_ = True
        return self

    def predict_proba(self, x) -> np.ndarray:
        """Возвращает вектор вероятностей класса 1 формы (N,)."""
        if not self.is_fitted_:
            raise RuntimeError("Сначала вызови fit.")

        x_tensor = self._prepare_x(x)

        if x_tensor.shape[1] != len(self.weights_):
            raise ValueError(
                "Количество признаков отличается от обучающей выборки."
            )

        logits = x_tensor @ self.weights_ + self.bias_
        probabilities = torch.sigmoid(logits)

        return probabilities.numpy()

    def predict(self, x) -> np.ndarray:
        probabilities = self.predict_proba(x)

        return (probabilities >= self.threshold).astype(np.int64)
