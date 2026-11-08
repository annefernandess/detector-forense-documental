"""
Diagnóstico temporário (só train): as imagens do Find it again! mostram a grade de 8 x 8 do JPEG?
Razão = média do gradiente nas bordas de bloco (colunas e linhas = 0 mod 8) / média nas demais, com
colunas e linhas juntas. Grade detectada se a razão >= 1,2 (fixado antes de rodar).
Rodar na raiz: python notebooks/diagnostico_grade_train.py
"""
import hashlib
import json
import sys
import warnings
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import cv2
import numpy as np

from dataset_loader import load_dataset, load_gray

LIMIAR = 1.2


def razoes(caminho):
    """(razão pooled, razão das colunas, razão das linhas) de uma imagem."""
    cv2.setNumThreads(1)
    g = load_gray(caminho).astype(np.float64)
    gh = np.abs(np.diff(g, axis=1)).mean(axis=0)   # gh[i]: borda entre as colunas i e i+1
    gv = np.abs(np.diff(g, axis=0)).mean(axis=1)
    fronteira = lambda n: (np.arange(1, n) % 8) == 0  # a borda de bloco fica antes da coluna 8k (entre 8k-1 e 8k)
    bh, bv = fronteira(len(gh) + 1), fronteira(len(gv) + 1)
    # np.diff índice i liga i e i+1; a coluna i+1 = 8k é borda, ou seja i+1 % 8 == 0
    mh, mv = (np.arange(len(gh)) + 1) % 8 == 0, (np.arange(len(gv)) + 1) % 8 == 0
    razao = lambda v, m: float(v[m].mean() / v[~m].mean())
    pooled = float(np.concatenate([gh[mh], gv[mv]]).mean() / np.concatenate([gh[~mh], gv[~mv]]).mean())
    return pooled, razao(gh, mh), razao(gv, mv)


def resumo(valores):
    v = np.array(valores)
    return {"n": int(len(v)), "fracao_detectada_pooled": float((v[:, 0] >= LIMIAR).mean()),
            "fracao_ambos_eixos": float(((v[:, 1] >= LIMIAR) & (v[:, 2] >= LIMIAR)).mean()),
            "fracao_algum_eixo": float(((v[:, 1] >= LIMIAR) | (v[:, 2] >= LIMIAR)).mean()),
            "razao_pooled_percentis_5_25_50_75_95": [float(x) for x in np.percentile(v[:, 0], [5, 25, 50, 75, 95])]}


if __name__ == "__main__":
    train = [s for s in load_dataset("data/findit2", "train") if s.image_path.exists()]
    aut = [str(s.image_path) for s in train if not s.is_tampered]
    fal = [str(s.image_path) for s in train if s.is_tampered]
    with get_context("fork").Pool(8) as pool:
        r_aut, r_fal = pool.map(razoes, aut), pool.map(razoes, fal)
    tabela = {"autenticas": resumo(r_aut), "falsificadas": resumo(r_fal)}
    parar = tabela["autenticas"]["fracao_detectada_pooled"] < 0.5
    saida = {"escopo": "só train do Find it again!", "limiar_razao": LIMIAR,
             "definicao": "razão pooled = média do gradiente em colunas e linhas = 0 mod 8 / média nas demais",
             "regra_de_parada": "parar se < 50% das autênticas tiverem grade detectada (razão pooled >= 1,2)",
             "tabela": tabela, "parar": bool(parar), "codigo": "notebooks/diagnostico_grade_train.py",
             "codigo_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path("results/diagnostico_grade_train.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(tabela, ensure_ascii=False, indent=1))
    print("parar:", parar, "| sha256:", saida["codigo_sha256"])
