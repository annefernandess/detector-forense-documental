"""
Rodada única no test do Find it again!: v2 congelado contra o candidato (v2 + copy_move_par, configuração
congelada). Antes de abrir o test, confere os hashes registrados em results/classificador_copy_move_cv_fixado.json.
`ensaio` roda o mesmo código com o val no lugar do test (treinando só no train) e grava fora de results/.
Rodar na raiz: python notebooks/teste_final_candidato.py [ensaio]
"""
import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classification import bootstrap_auc
from copy_move_feature import feature_lote
from dataset_loader import load_dataset
from features import load_or_extract
from features_v2 import load_or_extract_v2

ENSAIO = len(sys.argv) > 1 and sys.argv[1] == "ensaio"
SAIDA = Path("/private/tmp/claude-501/-Users-anne-Documents-UFPB-PDI/e66b19d1-edbb-47c8-a1f2-eab9280a9d21/scratchpad/ensaio_teste_final.json") if ENSAIO else Path("results/teste_final_candidato.json")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tabela(split, amostras):
    """Features v1 + v2 (cache) de um split, ordenadas como as amostras."""
    return (load_or_extract(amostras, f"data/features_{split}.csv")
            .merge(load_or_extract_v2(amostras, f"data/features_v2_{split}.csv"), on="image", validate="one_to_one"))


def ic_pareado(y, p_cand, p_v2, n_boot=2000, seed=42):
    """Diferença de AUC (candidato - v2) com IC de bootstrap sobre as imagens do test (mesmas reamostras para os dois)."""
    rng, difs = np.random.default_rng(seed), []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            difs.append(roc_auc_score(y[idx], p_cand[idx]) - roc_auc_score(y[idx], p_v2[idx]))
    return float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5)), len(difs)


if __name__ == "__main__":
    if not ENSAIO and SAIDA.exists():
        sys.exit("results/teste_final_candidato.json já existe: a rodada única já foi feita. Não rodar de novo.")
    fixado = json.loads(Path("results/classificador_copy_move_cv_fixado.json").read_text(encoding="utf-8"))
    esperados = {**fixado["codigo"], **fixado["teste"], **fixado["reutiliza_sem_alterar"],
                 "results/copy_move_residuo_val.json": fixado["feature_nova"]["sha256_arquivo_da_configuracao"]}
    hashes = {arquivo: sha(arquivo) for arquivo in esperados}
    diferentes = [a for a in esperados if hashes[a] != esperados[a]]
    if diferentes:
        sys.exit(f"hash diferente do registrado, parando antes de abrir o test: {diferentes}")
    cfg = fixado["feature_nova"]["configuracao"]
    config = json.loads(Path("results/configuracao_congelada.json").read_text(encoding="utf-8"))
    colunas_v2, nova = config["features"], "copy_move_par"

    avaliacao = "val" if ENSAIO else "test"
    treino_splits = ("train",) if ENSAIO else ("train", "val")
    amostras = {s: load_dataset("data/findit2", s) for s in set(treino_splits) | {avaliacao}}
    copia = pd.read_csv("data/features_copy_move_cv.csv")[["image", nova]]  # feature de train e val, já calculada
    treino = pd.concat([tabela(s, amostras[s]).merge(copia, on="image", validate="one_to_one") for s in treino_splits], ignore_index=True)

    S = amostras[avaliacao]
    feature = feature_lote([s.image_path for s in S], cfg["piso_intensidade"], cfg["k"], cfg["limiar_residuo"])  # mesmo código de train/val
    teste = tabela(avaliacao, S).merge(pd.DataFrame({"image": [s.image_path.name for s in S], nova: [feature[str(s.image_path)] for s in S]}),
                                       on="image", validate="one_to_one")
    assert not teste[nova].isna().any()

    ytr, yte = treino["is_tampered"].to_numpy(), teste["is_tampered"].to_numpy()
    modelo = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    v2 = modelo().fit(treino[colunas_v2].to_numpy(), ytr) if ENSAIO else joblib.load("results/modelo_v2.joblib")
    candidato = modelo().fit(treino[colunas_v2 + [nova]].to_numpy(), ytr)
    p_v2 = v2.predict_proba(teste[colunas_v2].to_numpy())[:, 1]
    p_cand = candidato.predict_proba(teste[colunas_v2 + [nova]].to_numpy())[:, 1]

    auc_v2, auc_cand = float(roc_auc_score(yte, p_v2)), float(roc_auc_score(yte, p_cand))
    lo, hi, n_validas = ic_pareado(yte, p_cand, p_v2)
    ic_v2, ic_cand = bootstrap_auc(yte, p_v2), bootstrap_auc(yte, p_cand)
    atingido = bool(auc_cand > auc_v2 and lo > 0)
    saida = {"rodada": "ensaio com val (não é resultado)" if ENSAIO else "única no test do Find it again!",
             "executado_em": datetime.now().isoformat(timespec="seconds"),
             "hashes_conferidos": hashes, "modelo_v2": "results/modelo_v2.joblib (congelado)" if not ENSAIO else "v2 refeito só no train (ensaio)",
             "modelo_v2_sha256": None if ENSAIO else sha("results/modelo_v2.joblib"),
             "configuracao_copy_move": cfg, "n_treino": int(len(treino)), "n_avaliacao": int(len(teste)), "n_adulteradas": int(yte.sum()),
             "feature": {"adulteradas_com_par": int(teste[nova][yte == 1].sum()), "autenticas_com_par": int(teste[nova][yte == 0].sum()),
                         "fracao_adulteradas": float(teste[nova][yte == 1].mean()), "fracao_autenticas": float(teste[nova][yte == 0].mean()),
                         "auc_isolada": float(roc_auc_score(yte, teste[nova]))},
             "auc_v2": auc_v2, "auc_v2_ic_bootstrap": list(ic_v2), "auc_candidato": auc_cand, "auc_candidato_ic_bootstrap": list(ic_cand),
             "diferenca_auc": auc_cand - auc_v2, "diferenca_ic_bootstrap_2000_seed42": [lo, hi], "reamostras_validas": n_validas,
             "criterio": "AUC do candidato > AUC do v2 no test E limite inferior do IC pareado > 0", "criterio_atingido": atingido,
             "decisao": "candidato vira o classificador final" if atingido else "v2 continua como classificador final; a feature fica só como evidência de localização"}
    SAIDA.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: saida[k] for k in ("rodada", "n_avaliacao", "n_adulteradas", "feature", "auc_v2", "auc_v2_ic_bootstrap", "auc_candidato",
                                            "auc_candidato_ic_bootstrap", "diferenca_auc", "diferenca_ic_bootstrap_2000_seed42", "criterio_atingido", "decisao")},
                     ensure_ascii=False, indent=1))
