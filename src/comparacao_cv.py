"""
AUC por partição da validação cruzada do classificador (mesmo divisor e mesma semente de
classification.cv_evaluate, que só devolve médias) e comparação pareada entre dois conjuntos de
features nas mesmas partições.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


def auc_por_particao(make_model, X: np.ndarray, y: np.ndarray, n_splits: int = 5, n_repeats: int = 3,
                     seed: int = 42) -> np.ndarray:
    """AUC no fold de validação de cada partição (n_splits x n_repeats valores), na ordem de cv_evaluate."""
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return np.array([roc_auc_score(y[te], make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
                     for tr, te in splitter.split(X, y)])


def diferenca_pareada(auc_a: np.ndarray, auc_b: np.ndarray, n_splits: int = 5) -> dict:
    """
    Diferença média (a - b) nas mesmas partições, com IC 95%. O IC principal usa a correção de Nadeau e Bengio
    para validação cruzada repetida (variância x (1/k + n_teste/n_treino), com n_teste/n_treino = 1/(n_splits-1)),
    porque as partições compartilham dados e o IC ingênuo é estreito demais. O IC ingênuo (t de Student sobre as
    partições) vai junto, só para comparação.
    """
    d = np.asarray(auc_a) - np.asarray(auc_b)
    k, media, variancia = len(d), float(d.mean()), float(d.var(ddof=1))
    t = stats.t.ppf(0.975, k - 1)
    erro_corrigido = np.sqrt((1 / k + 1 / (n_splits - 1)) * variancia)
    erro_ingenuo = np.sqrt(variancia / k)
    return {"diferenca_media": media, "desvio": float(np.sqrt(variancia)), "n_particoes": k,
            "ic95_corrigido": [media - t * erro_corrigido, media + t * erro_corrigido],
            "ic95_ingenuo": [media - t * erro_ingenuo, media + t * erro_ingenuo]}
