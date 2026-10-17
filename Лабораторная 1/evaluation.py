import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Ожидаются одномерные массивы.")

    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError(
            "Массивы должны иметь одинаковую ненулевую длину."
        )

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
    ):
        raise ValueError("Массивы содержат NaN или inf.")

    errors = predicted - actual

    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)

    if ss_tot == 0:
        raise ValueError(
            "R² не определён для постоянной целевой переменной."
        )

    r2 = 1.0 - ss_res / ss_tot

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }

def classification_metrics(
    y_true,
    y_proba,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    actual = np.asarray(y_true, dtype=np.float64)
    probabilities = np.asarray(y_proba, dtype=np.float64)

    if actual.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Ожидаются одномерные массивы.")

    if actual.shape != probabilities.shape or actual.size == 0:
        raise ValueError(
            "Массивы должны иметь одинаковую ненулевую длину."
        )

    if not np.isin(actual, [0, 1]).all():
        raise ValueError("y_true должен содержать только 0 и 1.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Вероятности содержат NaN или inf.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Вероятности должны находиться от 0 до 1.")

    if not 0 < threshold < 1:
        raise ValueError("threshold должен находиться между 0 и 1.")

    predicted = (probabilities >= threshold).astype(np.int64)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())

    accuracy = (tp + tn) / len(actual)

    # При нулевом знаменателе принимаем значение метрики за 0.
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn > 0
        else 0.0
    )

    safe_probabilities = np.clip(
        probabilities,
        1e-15,
        1 - 1e-15,
    )

    logloss = -np.mean(
        actual * np.log(safe_probabilities)
        + (1 - actual) * np.log1p(-safe_probabilities)
    )

    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "LogLoss": float(logloss),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }
