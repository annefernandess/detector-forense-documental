"""Testes com assert para a validação cruzada. Rodar: python tests/test_classification.py"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from sklearn.linear_model import LogisticRegression

from classification import bootstrap_auc, cv_evaluate, nested_cv

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
rng = np.random.default_rng(42)
y = np.array([0] * 160 + [1] * 40)
make = lambda: LogisticRegression(class_weight="balanced", max_iter=1000)

# Sinal forte: a feature separa as classes, a AUC fica perto de 1.
X_sinal = (y + rng.normal(0, 0.3, len(y)))[:, None]
assert cv_evaluate(make, X_sinal, y)["auc_mean"] > 0.95

# Sem sinal: a feature é ruído, a AUC fica perto de 0,5.
r = cv_evaluate(make, rng.normal(0, 1, (len(y), 1)), y)
assert 0.35 < r["auc_mean"] < 0.65 and r["n_folds"] == 15, r

# Validação aninhada: entre uma configuração informativa e uma de ruído, a interna
# escolhe a informativa na maioria dos folds, e a AUC da escolha fica alta.
res = nested_cv({"sinal": (X_sinal, make), "ruido": (rng.normal(0, 1, (len(y), 1)), make)}, y)
assert res["chosen_counts"].get("sinal", 0) >= 13 and res["chosen_auc_mean"] > 0.9, res

# Bootstrap da AUC: separação perfeita dá IC (1, 1); probabilidades aleatórias dão IC em torno de 0,5.
assert bootstrap_auc(y, y.astype(float), n_boot=200) == (1.0, 1.0)
lo, hi = bootstrap_auc(y, rng.random(len(y)), n_boot=500)
assert lo < 0.5 < hi, (lo, hi)

print("ok")
