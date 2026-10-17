import numpy as np
import pandas as pd


class BikePreprocessor:
    """Стандартизация числовых и one-hot кодирование категорий."""

    def __init__(
        self,
        numeric_columns: list[str],
        binary_columns: list[str],
        categorical_columns: list[str],
    ):
        self.numeric_columns = list(numeric_columns)
        self.binary_columns = list(binary_columns)
        self.categorical_columns = list(categorical_columns)

        self.columns = (
            self.numeric_columns
            + self.binary_columns
            + self.categorical_columns
        )

        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Группы признаков не должны пересекаться.")

        self.mean_ = np.empty(0, dtype=np.float64)
        self.scale_ = np.empty(0, dtype=np.float64)
        self.categories_: dict[str, list] = {}
        self.feature_names_: list[str] = []
        self.is_fitted_ = False

    def _check_input(self, x: pd.DataFrame):
        missing_columns = set(self.columns) - set(x.columns)

        if missing_columns:
            raise ValueError(
                f"Отсутствуют столбцы: {sorted(missing_columns)}"
            )

        if x[self.columns].isna().to_numpy().any():
            raise ValueError(
                "BikePreprocessor ожидает данные без пропусков."
            )

        values = x[
            self.numeric_columns + self.binary_columns
        ].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError("Обнаружены бесконечные значения.")

        for column in self.binary_columns:
            if not x[column].isin([0, 1]).all():
                raise ValueError(
                    f"Признак {column} должен содержать только 0 и 1."
                )

    def fit(self, x: pd.DataFrame):
        self.is_fitted_ = False
        self._check_input(x)

        if len(x) == 0:
            raise ValueError("Обучающая выборка пуста.")

        numeric = x[self.numeric_columns].to_numpy(
            dtype=np.float64
        )

        self.mean_ = numeric.mean(axis=0)
        self.scale_ = numeric.std(axis=0, ddof=0)

        # Постоянный признак не должен приводить к делению на ноль.
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)

        self.categories_ = {
            column: sorted(x[column].unique().tolist())
            for column in self.categorical_columns
        }

        self.feature_names_ = (
            self.numeric_columns + self.binary_columns
        )

        for column in self.categorical_columns:
            self.feature_names_.extend(
                f"{column}={category}"
                for category in self.categories_[column]
            )

        self.is_fitted_ = True
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Сначала вызови fit на train.")

        self._check_input(x)

        numeric = x[self.numeric_columns].to_numpy(
            dtype=np.float64
        )
        numeric = (numeric - self.mean_) / self.scale_

        binary = x[self.binary_columns].to_numpy(
            dtype=np.float64
        )

        blocks = [numeric, binary]

        for column in self.categorical_columns:
            values = x[column].to_numpy()

            for category in self.categories_[column]:
                encoded = (values == category).astype(np.float64)
                blocks.append(encoded.reshape(-1, 1))

        return np.concatenate(blocks, axis=1)

    def fit_transform(self, x: pd.DataFrame) -> np.ndarray:
        return self.fit(x).transform(x)


class SpaceshipPreprocessor:
    """Подготовка признаков Spaceship Titanic без sklearn."""

    def __init__(self):
        self.spending_columns = [
            "RoomService",
            "FoodCourt",
            "ShoppingMall",
            "Spa",
            "VRDeck",
        ]

        self.numeric_columns = [
            "Age",
            "CabinNumber",
        ] + self.spending_columns

        self.categorical_columns = [
            "HomePlanet",
            "Destination",
            "CryoSleep",
            "VIP",
            "CabinDeck",
            "CabinSide",
        ]

        self.required_columns = [
            "Age",
            "Cabin",
            "HomePlanet",
            "Destination",
            "CryoSleep",
            "VIP",
        ] + self.spending_columns

        self.medians_ = pd.Series(dtype=np.float64)
        self.mean_ = np.empty(0, dtype=np.float64)
        self.scale_ = np.empty(0, dtype=np.float64)
        self.categories_: dict[str, list[str]] = {}
        self.feature_names_: list[str] = []
        self.is_fitted_ = False

    def _prepare_frame(self, x: pd.DataFrame) -> pd.DataFrame:
        missing_columns = set(self.required_columns) - set(x.columns)

        if missing_columns:
            raise ValueError(
                f"Отсутствуют столбцы: {sorted(missing_columns)}"
            )

        frame = x[self.required_columns].copy()

        # Каждая строка преобразуется независимо от остальных.
        cabin_parts = (
            frame["Cabin"]
            .astype("string")
            .str.split("/", expand=True)
            .reindex(columns=[0, 1, 2])
        )

        frame["CabinDeck"] = cabin_parts[0]
        frame["CabinNumber"] = pd.to_numeric(
            cabin_parts[1],
            errors="coerce",
        )
        frame["CabinSide"] = cabin_parts[2]

        for column in self.numeric_columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="raise",
            ).astype(np.float64)

        numeric = frame[self.numeric_columns].to_numpy(
            dtype=np.float64
        )

        if np.isinf(numeric).any():
            raise ValueError("Числовые признаки содержат inf.")

        if (frame[self.spending_columns] < 0).to_numpy().any():
            raise ValueError("Расходы не могут быть отрицательными.")

        # True, False и пропуск становятся строковыми категориями.
        for column in self.categorical_columns:
            frame[column] = (
                frame[column]
                .astype("string")
                .fillna("Unknown")
            )

        return frame

    def _numeric_values(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = (
            frame[self.numeric_columns]
            .fillna(self.medians_)
            .to_numpy(dtype=np.float64, copy=True)
        )

        # Первые два столбца — Age и CabinNumber.
        # Остальные пять — расходы.
        numeric[:, 2:] = np.log1p(numeric[:, 2:])

        return numeric

    def fit(self, x: pd.DataFrame):
        self.is_fitted_ = False

        if len(x) == 0:
            raise ValueError("Обучающая выборка пуста.")

        frame = self._prepare_frame(x)

        self.medians_ = frame[self.numeric_columns].median()

        if self.medians_.isna().any():
            raise ValueError(
                "Нельзя вычислить медиану: числовой столбец "
                "целиком состоит из пропусков."
            )

        numeric = self._numeric_values(frame)

        self.mean_ = numeric.mean(axis=0)
        self.scale_ = numeric.std(axis=0, ddof=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)

        self.categories_ = {}

        for column in self.categorical_columns:
            observed = set(frame[column].tolist())

            # Резервируем категорию для пропусков и новых значений.
            observed.add("Unknown")
            self.categories_[column] = sorted(observed)

        self.feature_names_ = list(self.numeric_columns)

        for column in self.categorical_columns:
            self.feature_names_.extend(
                f"{column}={category}"
                for category in self.categories_[column]
            )

        self.is_fitted_ = True
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Сначала вызови fit на train.")

        frame = self._prepare_frame(x)

        numeric = self._numeric_values(frame)
        numeric = (numeric - self.mean_) / self.scale_

        blocks = [numeric]

        for column in self.categorical_columns:
            categories = self.categories_[column]

            values = frame[column].where(
                frame[column].isin(categories),
                "Unknown",
            ).to_numpy(dtype=str)

            for category in categories:
                encoded = (values == category).astype(np.float64)
                blocks.append(encoded.reshape(-1, 1))

        result = np.concatenate(blocks, axis=1)

        if not np.isfinite(result).all():
            raise ValueError(
                "После преобразования обнаружены NaN или inf."
            )

        return result

    def fit_transform(self, x: pd.DataFrame) -> np.ndarray:
        return self.fit(x).transform(x)