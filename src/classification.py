"""Validação cruzada para o classificador de imagem (notebook 04; a validação aninhada e o
bootstrap foram usados em notebooks/arquivo/06_alinhamento_sem_ganho.ipynb)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from evaluation import youden_threshold


def cv_evaluate(make_model, X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                n_repeats: int = 3, seed: int = 42) -> dict:
    """
    Validação cruzada estratificada repetida. Em cada fold, o modelo é treinado nos folds
    de treino, o limiar de Youden é calculado nas probabilidades dos próprios folds de
    treino, e AUC, sensibilidade e especificidade são medidas no fold de validação.
    Retorna médias, desvio da AUC entre folds e o limiar médio.
    """
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    aucs, sens, spec, thresholds = [], [], [], []
    for train_idx, test_idx in splitter.split(X, y):
        model = make_model().fit(X[train_idx], y[train_idx])
        p_train = model.predict_proba(X[train_idx])[:, 1]
        p_test = model.predict_proba(X[test_idx])[:, 1]
        yt = y[train_idx]
        threshold, _, _ = youden_threshold(p_train[yt == 1], p_train[yt == 0], np.unique(p_train))
        pred, truth = p_test >= threshold, y[test_idx]
        aucs.append(roc_auc_score(truth, p_test))
        sens.append(pred[truth == 1].mean())
        spec.append((~pred[truth == 0]).mean())
        thresholds.append(threshold)
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "sens_mean": float(np.mean(sens)), "spec_mean": float(np.mean(spec)),
            "threshold": float(np.mean(thresholds)), "n_folds": len(aucs)}


def nested_cv(configs: dict, y: np.ndarray, outer_splits: int = 5, outer_repeats: int = 3,
              inner_splits: int = 5, seed: int = 42) -> dict:
    """
    Validação cruzada aninhada. `configs` mapeia nome -> (X, fábrica do modelo). Em cada
    fold externo: (1) toda configuração é treinada nos folds de treino e medida no fold
    externo (AUC externa por configuração, e limiar de Youden nas probabilidades de
    treino); (2) uma validação interna, só nos folds de treino, escolhe a configuração de
    maior AUC (empate: menos features), e registra-se a AUC externa dessa escolha.
    """
    from sklearn.model_selection import StratifiedKFold

    names = list(configs)
    n_rows = len(y)
    outer = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=seed)
    auc = {n: [] for n in names}
    thr = {n: [] for n in names}
    chosen_auc, chosen = [], []
    for train_idx, test_idx in outer.split(np.zeros(n_rows), y):
        yt = y[train_idx]
        for n in names:
            X, make = configs[n]
            model = make().fit(X[train_idx], yt)
            p_train = model.predict_proba(X[train_idx])[:, 1]
            auc[n].append(roc_auc_score(y[test_idx], model.predict_proba(X[test_idx])[:, 1]))
            thr[n].append(youden_threshold(p_train[yt == 1], p_train[yt == 0], np.unique(p_train))[0])

        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        inner_auc = {n: [] for n in names}
        for in_tr, in_te in inner.split(np.zeros(len(train_idx)), yt):
            for n in names:
                X, make = configs[n]
                Xo = X[train_idx]
                m = make().fit(Xo[in_tr], yt[in_tr])
                inner_auc[n].append(roc_auc_score(yt[in_te], m.predict_proba(Xo[in_te])[:, 1]))
        best = max(names, key=lambda n: (np.mean(inner_auc[n]), -configs[n][0].shape[1]))
        chosen.append(best)
        chosen_auc.append(auc[best][-1])

    return {
        "per_config": {n: {"auc_mean": float(np.mean(auc[n])), "auc_std": float(np.std(auc[n])),
                           "threshold": float(np.mean(thr[n])), "n_features": configs[n][0].shape[1]}
                       for n in names},
        "chosen_auc_mean": float(np.mean(chosen_auc)), "chosen_auc_std": float(np.std(chosen_auc)),
        "chosen_counts": {n: chosen.count(n) for n in names if chosen.count(n)},
    }


def bootstrap_auc(y: np.ndarray, prob: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """IC 95% da AUC por bootstrap: reamostra as imagens com reposição `n_boot` vezes e
    toma os percentis 2,5 e 97,5. Reamostras com uma só classe são descartadas."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], prob[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
