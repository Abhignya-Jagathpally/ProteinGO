"""Protein-centric Fmax, Smin, AUPR, AUC, and MCC."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)


THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 3)


def project_scores(probs: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    out = np.asarray(probs, dtype=np.float32).copy()
    for child, parent in edges:
        out[:, parent] = np.maximum(out[:, parent], out[:, child])
    return out


def protein_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    hit = (y_true * y_pred).sum(axis=1)
    positives = y_true.sum(axis=1)
    guesses = y_pred.sum(axis=1)
    has_pred = guesses > 0
    has_true = positives > 0
    precision = float(np.mean(hit[has_pred] / guesses[has_pred])) if has_pred.any() else 0.0
    recall = float(np.mean(hit[has_true] / positives[has_true])) if has_true.any() else 0.0
    return 2 * precision * recall / (precision + recall + 1e-12)


def semantic_distance(y_true, y_pred, information) -> float:
    remaining = float(
        np.mean(((y_true == 1) & (y_pred == 0)) * information[None, :], axis=0).sum()
    )
    misinfo = float(
        np.mean(((y_true == 0) & (y_pred == 1)) * information[None, :], axis=0).sum()
    )
    return float(np.hypot(remaining, misinfo))


def _safe_aupr(y_true, probs) -> float:
    try:
        return float(average_precision_score(y_true, probs, average="micro"))
    except ValueError:
        return float("nan")


def _safe_auc(y_true, probs) -> float:
    try:
        return float(roc_auc_score(y_true, probs, average="micro"))
    except ValueError:
        return float("nan")


def evaluate_split(name, y_true, probs, edges, information, namespaces=None) -> dict:
    probs = project_scores(probs, edges)
    f_values = np.array([protein_f1(y_true, probs >= threshold) for threshold in THRESHOLDS])
    s_values = np.array(
        [
            semantic_distance(y_true, probs >= threshold, information)
            for threshold in THRESHOLDS
        ]
    )
    best_f = float(THRESHOLDS[int(np.argmax(f_values))])
    best_s = float(THRESHOLDS[int(np.argmin(s_values))])
    labels = probs >= best_f
    try:
        mcc = float(matthews_corrcoef(y_true.ravel().astype(int), labels.ravel().astype(int)))
    except ValueError:
        mcc = float("nan")
    result = {
        "split": name,
        "Fmax": float(f_values.max()),
        "Smin": float(s_values.min()),
        "F_threshold": best_f,
        "S_threshold": best_s,
        "AUPR": _safe_aupr(y_true, probs),
        "AUC": _safe_auc(y_true, probs),
        "MCC": mcc,
        "n": int(len(y_true)),
    }
    print(
        f"[eval] {name} n={result['n']} Fmax={result['Fmax']:.4f} "
        f"Smin={result['Smin']:.4f} AUPR={result['AUPR']:.4f} "
        f"AUC={result['AUC']:.4f} MCC={result['MCC']:.4f} "
        f"F_threshold={best_f:.2f}"
    )
    if namespaces is not None:
        for namespace in ("molecular_function", "biological_process", "cellular_component"):
            columns = [index for index, item in enumerate(namespaces) if item == namespace]
            if not columns:
                continue
            part_true = y_true[:, columns]
            part_probs = probs[:, columns]
            part_f = max(
                protein_f1(part_true, part_probs >= threshold) for threshold in THRESHOLDS
            )
            print(f"[eval] {name} {namespace} terms={len(columns)} Fmax={part_f:.4f}")
            result[f"Fmax_{namespace}"] = float(part_f)
    return result
