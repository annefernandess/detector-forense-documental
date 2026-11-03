"""
Avaliação do modelo congelado (v2) no conjunto sintético de copy-move (data/cord_sintetico/).
Nada é treinado nem recalibrado: usa results/modelo_v2.joblib e o limiar de
results/configuracao_congelada.json. A caixa colada, registrada em inventario.csv, serve de
gabarito só para medir a localização; as features não a usam.
"""

from __future__ import annotations

import csv
import functools
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from classification import bootstrap_auc
from copy_move import blind_copy_move_for_path, copy_move_region_score, region_covered
from dataset_loader import DocumentSample, TamperRegion
from ela import compute_ela, compute_region_score
from evaluation import evaluate_region_technique, wilson_score_interval
from features import extract_features
from features_v2 import component_features
from pipeline import LIMIAR_COPY_MOVE, carregar_congelado

LIMIAR_CORRELACAO = 0.979  # pico de correlação com caixa conhecida, escolhido no val (notebook 03)
COBERTURA_MIN = 0.5        # fração da região coberta por uma caixa da busca cega (region_covered)


def carregar_amostras(pasta: Path) -> list:
    """DocumentSample de cada linha de inventario.csv; a caixa colada vira região modificada (CPI)."""
    amostras = []
    with open(pasta / "inventario.csv", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            adulterada = linha["classe"] == "adulterada"
            caixa = tuple(int(linha[k]) for k in ("x", "y", "largura", "altura")) if adulterada else None
            regioes = [TamperRegion(caixa, "CPI", None, False)] if adulterada else []
            amostras.append(DocumentSample(pasta / linha["arquivo"], adulterada,
                                           ["CPI"] if adulterada else [], regioes))
    return amostras


def _features(caminho) -> dict:
    cv2.setNumThreads(1)
    return {**extract_features(caminho), **component_features(caminho)}


def pontuar(amostras: list, processos: int = 8):
    """(features por imagem, probabilidades do modelo congelado, limiar congelado)."""
    modelo, config = carregar_congelado()
    with get_context("fork").Pool(processos) as pool:
        valores = pool.map(_features, [str(s.image_path) for s in amostras])
    x = np.array([[v[nome] for nome in config["features"]] for v in valores])
    return valores, modelo.predict_proba(x)[:, 1], config["limiar"]


def metricas_classificacao(y: np.ndarray, prob: np.ndarray, limiar: float) -> dict:
    """Sensibilidade e falso alarme no limiar congelado (Wilson 95%) e AUC (bootstrap 95%)."""
    marcada = prob >= limiar
    tp, fp = int(marcada[y == 1].sum()), int(marcada[y == 0].sum())
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return {"limiar": limiar, "n_adulteradas": n1, "n_autenticas": n0,
            "sensibilidade": tp / n1, "sensibilidade_acertos": tp, "sensibilidade_wilson": wilson_score_interval(tp / n1, n1),
            "falso_alarme": fp / n0, "falso_alarme_marcadas": fp, "falso_alarme_wilson": wilson_score_interval(fp / n0, n0),
            "auc": float(roc_auc_score(y, prob)), "auc_ic_bootstrap": bootstrap_auc(y, prob)}


@functools.lru_cache(maxsize=4)
def _mapa_ela(caminho):
    return compute_ela(caminho, quality=90, amplification=1, grayscale=True)[0]


def ela_score(caminho, regioes):
    """Score no formato de evaluate_region_technique (média da ELA na região, média no controle)."""
    return compute_region_score(caminho, _mapa_ela(str(caminho)), regioes)


@functools.lru_cache(maxsize=4)
def _caixas_cegas(caminho):
    return tuple(blind_copy_move_for_path(caminho, LIMIAR_COPY_MOVE)[1])


def busca_cega_score(caminho, regioes):
    """1.0 se a busca cega cobre (>= COBERTURA_MIN) alguma região modificada, senão 0.0; o segundo
    valor é só preenchimento exigido pelo formato (a regra de acerto usa o primeiro)."""
    alvos = [r for r in regioes if r.is_original_area is not True]
    caixas = list(_caixas_cegas(str(caminho)))
    return float(any(region_covered(r.bbox, caixas, COBERTURA_MIN) for r in alvos)), 0.0


def localizacao(amostras: list) -> dict:
    """Três medidas de localização contra a caixa registrada, cada uma com pseudo-regiões ao acaso
    (evaluate_region_technique, seed 42): ELA (região > controle), pico de correlação com a caixa
    conhecida (>= LIMIAR_CORRELACAO) e busca cega (caixa detectada cobre a região)."""
    return {
        "ela": evaluate_region_technique(amostras, ela_score, lambda sin, sout: sin > sout),
        "copy_move_caixa_conhecida": evaluate_region_technique(
            amostras, copy_move_region_score, lambda sin, sout: sin >= LIMIAR_CORRELACAO),
        "copy_move_busca_cega": evaluate_region_technique(
            amostras, busca_cega_score, lambda sin, sout: sin >= 1.0),
    }
