"""
Congela o candidato (v2 + copy_move_par) pela receita de notebooks/teste_final_candidato.py: 13 features na ordem
usada, StandardScaler + LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42), treinada
nas 769 imagens de train + val. Não abre o test. Etapas: determinismo em dois processos, limiar de Youden nas
previsões fora da amostra da validação cruzada 5 x 3 (seed 42), salvar o modelo e gravar a configuração.
Rodar na raiz: python notebooks/congelar_candidato.py
"""
import hashlib
import json
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset_loader import load_dataset
from evaluation import youden_threshold
from features import load_or_extract
from features_v2 import load_or_extract_v2

TMP = Path("/private/tmp/claude-501/-Users-anne-Documents-UFPB-PDI/e66b19d1-edbb-47c8-a1f2-eab9280a9d21/scratchpad")
NOVA = "copy_move_par"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dados():
    """Tabela de train + val (nessa ordem, como em teste_final_candidato.py), colunas das 13 features e rótulo."""
    config = json.loads(Path("results/configuracao_congelada.json").read_text(encoding="utf-8"))
    copia = pd.read_csv("data/features_copy_move_cv.csv")[["image", NOVA]]
    partes = []
    for s in ("train", "val"):
        S = load_dataset("data/findit2", s)
        t = (load_or_extract(S, f"data/features_{s}.csv")
             .merge(load_or_extract_v2(S, f"data/features_v2_{s}.csv"), on="image", validate="one_to_one"))
        partes.append(t.merge(copia, on="image", validate="one_to_one"))
    return pd.concat(partes, ignore_index=True), config["features"] + [NOVA]


modelo = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))

if __name__ == "__main__":
    tabela, colunas = dados()
    X, y = tabela[colunas].to_numpy(), tabela["is_tampered"].to_numpy()
    if len(sys.argv) > 2 and sys.argv[1] == "sonda":  # processo separado: treina a receita e guarda as probabilidades
        np.save(sys.argv[2], modelo().fit(X, y).predict_proba(X)[:, 1])
        sys.exit(0)

    assert len(tabela) == 769 and int(y.sum()) == 127 and len(colunas) == 13
    # Etapa 1: determinismo, em dois processos separados
    saidas = [TMP / f"sonda_{i}.npy" for i in (1, 2)]
    for s in saidas:
        subprocess.run([sys.executable, "-I", __file__, "sonda", str(s)], check=True)
    p1, p2 = (np.load(s) for s in saidas)
    n_train = int((tabela.index < len(load_dataset("data/findit2", "train"))).sum())
    determinismo = {"execucoes": 2, "processos_separados": True, "max_diff_train": float(np.abs(p1[:n_train] - p2[:n_train]).max()),
                    "max_diff_train_val": float(np.abs(p1 - p2).max()), "n_imagens_train": n_train}
    print("determinismo:", determinismo)
    if determinismo["max_diff_train"] >= 1e-12:
        sys.exit("diferença de determinismo não é zero: parando")

    # Etapa 2: limiar de Youden nas previsões fora da amostra da validação cruzada (5 x 3, seed 42)
    oof, rotulos, aucs = [], [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, y):
        p = modelo().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oof.append(p); rotulos.append(y[te]); aucs.append(roc_auc_score(y[te], p))
    oof, rotulos = np.concatenate(oof), np.concatenate(rotulos)
    limiar, tpr, fpr = youden_threshold(oof[rotulos == 1], oof[rotulos == 0], np.unique(oof))
    limiar_info = {"valor": limiar, "sensibilidade": tpr, "especificidade": 1 - fpr, "youden_J": tpr - fpr, "n_previsoes": int(len(oof)),
                   "n_previsoes_adulteradas": int((rotulos == 1).sum()), "auc_externa_media_das_particoes": float(np.mean(aucs)),
                   "origem": "previsões fora da amostra da validação cruzada 5 x 3 (seed 42) nas 769 imagens de train + val, 3 por imagem, agrupadas; maior índice de Youden (decide positivo se prob >= limiar); sem test e sem o v2"}
    print("limiar:", limiar_info)

    # Etapa 3: treinar a receita uma última vez e salvar
    final = modelo().fit(X, y)
    destino = Path("results/modelo_candidato.joblib")
    assert not destino.exists(), "results/modelo_candidato.joblib já existe"
    joblib.dump(final, destino)
    hash_modelo = sha(destino)
    carregado = joblib.load(destino)
    diff_salvo = float(np.abs(carregado.predict_proba(X)[:, 1] - p1).max())
    print("sha256 do modelo:", hash_modelo, "| diferença do modelo salvo contra a sonda 1:", diff_salvo)

    # Etapa 4: configuração congelada
    teste = json.loads(Path("results/teste_final_candidato.json").read_text(encoding="utf-8"))
    fixado = json.loads(Path("results/classificador_copy_move_cv_fixado.json").read_text(encoding="utf-8"))
    config = {"versao": "candidato (v2 + copy_move_par)", "congelado_em": datetime.now().isoformat(timespec="seconds"),
              "modelo": {"arquivo": "results/modelo_candidato.joblib", "sha256": hash_modelo,
                         "receita": "StandardScaler + LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42), treinada nas 769 imagens de train + val",
                         "versoes": {"scikit-learn": sklearn.__version__, "numpy": np.__version__}},
              "sha256_src_copy_move_residuo": sha("src/copy_move_residuo.py"), "sha256_src_copy_move_feature": sha("src/copy_move_feature.py"),
              "sha256_script_da_receita": sha("notebooks/teste_final_candidato.py"),
              "configuracao_tecnica_1": fixado["feature_nova"]["configuracao"],
              "features": colunas, "limiar": limiar_info,
              "determinismo": {**determinismo, "max_diff_modelo_salvo_contra_sonda": diff_salvo},
              "teste": {"auc_v2": teste["auc_v2"], "auc_v2_ic_bootstrap": teste["auc_v2_ic_bootstrap"],
                        "auc_candidato": teste["auc_candidato"], "auc_candidato_ic_bootstrap": teste["auc_candidato_ic_bootstrap"],
                        "diferenca_auc": teste["diferenca_auc"], "diferenca_ic_bootstrap_2000_seed42": teste["diferenca_ic_bootstrap_2000_seed42"],
                        "n_avaliacao": teste["n_avaliacao"], "n_adulteradas": teste["n_adulteradas"], "feature": teste["feature"],
                        "fonte": "results/teste_final_candidato.json (executado em " + teste["executado_em"] + ")"},
              "nota_sobre_o_teste": "O AUC do test foi calculado com um modelo treinado na rodada de 6/11 e não salvo. O modelo salvo nesta configuração é a reprodução da receita, com equivalência verificada por determinismo no train, e não o objeto avaliado no test.",
              "v2_de_referencia": "results/modelo_v2.joblib e results/configuracao_congelada.json (não alterados)"}
    Path("results/configuracao_candidato.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("gravado results/configuracao_candidato.json")
