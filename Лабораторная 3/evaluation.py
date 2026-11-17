"""Ручные бинарные метрики, ROC и PR без готовых реализаций метрик."""

import numpy as np


def _binary(y):
    labels = np.asarray(y)
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Метки должны быть непустым вектором из 0 и 1")
    return labels.astype(np.int64)


def confusion_matrix(y_true, y_pred):
    actual, predicted = _binary(y_true), _binary(y_pred)
    if actual.shape != predicted.shape:
        raise ValueError("Размеры векторов должны совпадать")
    return np.array([[np.sum((actual == 0) & (predicted == 0)),
                      np.sum((actual == 0) & (predicted == 1))],
                     [np.sum((actual == 1) & (predicted == 0)),
                      np.sum((actual == 1) & (predicted == 1))]], dtype=np.int64)


def classification_metrics(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"Accuracy": (tp + tn) / matrix.sum(), "Precision": precision,
            "Recall": recall, "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def _threshold_counts(y_true, scores):
    actual = _binary(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != actual.shape or not np.isfinite(scores).all():
        raise ValueError("Для каждого объекта нужен конечный score")
    positives = int(actual.sum())
    negatives = len(actual) - positives
    if not positives or not negatives:
        raise ValueError("Для ROC и PR нужны объекты обоих классов")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_y = actual[order]
    # Все объекты с одинаковым score переходят через порог одновременно.
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(actual) - 1]
    tp = np.r_[0, np.cumsum(sorted_y)[ends]]
    fp = np.r_[0, (ends + 1) - tp[1:]]
    thresholds = np.r_[np.inf, sorted_scores[ends]]
    return tp, fp, positives, negatives, thresholds


def roc_curve(y_true, scores):
    tp, fp, positives, negatives, thresholds = _threshold_counts(y_true, scores)
    return fp / negatives, tp / positives, thresholds


def precision_recall_curve(y_true, scores):
    tp, fp, positives, _, thresholds = _threshold_counts(y_true, scores)
    precision = np.divide(tp, tp + fp, out=np.ones(len(tp), dtype=float), where=(tp + fp) != 0)
    return tp / positives, precision, thresholds


def auc(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (x.ndim != 1 or len(x) < 2 or x.shape != y.shape or
            not np.isfinite(x).all() or not np.isfinite(y).all() or (np.diff(x) < 0).any()):
        raise ValueError("Кривая должна состоять из конечных точек с неубывающей x")
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2))
