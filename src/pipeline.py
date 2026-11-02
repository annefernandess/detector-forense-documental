"""
Análise de uma imagem avulsa para a interface. Carrega o modelo congelado da v2
(results/modelo_v2.joblib), calcula as 12 features pela mesma extração do lote
(`extract_features` e `component_features`) e devolve a faixa de decisão, a posição entre
as imagens de referência (val) e os dados das visualizações. Não treina, não recalibra,
não lê o test e não usa anotações.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from copy_move import blind_copy_move
from dataset_loader import load_gray
from ela import compute_ela
from features import extract_features
from explicacao import explicar
from features_v2 import component_features
from jpeg_ghost import DEFAULT_QUALITIES, compute_jpeg_ghost
from regioes import regioes_ela

ROOT = Path(__file__).resolve().parent.parent
RESULTADOS = ROOT / "results"
LARGURA_EXIBICAO = 1000
LIMIAR_COPY_MOVE = 0.99  # busca cega: limiar escolhido no val (notebook 03)

# Medidas do test da v2 (notebooks/04_classificador.ipynb, célula de avaliação). O test foi
# usado nas avaliações v1 e v2, então são estimativas otimistas.
METRICAS_TEST = {
    "adulteradas": 35, "autenticas": 183,
    "marcadas_adulteradas": 16, "marcadas_autenticas": 64,
    "sensibilidade": 0.457, "ic_sensibilidade": (0.305, 0.618),
    "precisao": 0.200, "ic_precisao": (0.127, 0.300),
    "especificidade": 0.650, "auc": 0.626, "acuracia_tudo_autentico": 0.839,
}


@functools.lru_cache(maxsize=1)
def carregar_congelado():
    """(modelo, configuração) lidos de results/modelo_v2.joblib e configuracao_congelada.json."""
    config = json.loads((RESULTADOS / "configuracao_congelada.json").read_text(encoding="utf-8"))
    return joblib.load(RESULTADOS / "modelo_v2.joblib"), config


@functools.lru_cache(maxsize=1)
def carregar_regioes_config() -> dict:
    """Parâmetros das caixas automáticas e medidas da calibração, de results/regioes_config.json
    (calibrado no val)."""
    return json.loads((RESULTADOS / "regioes_config.json").read_text(encoding="utf-8"))


def pontuacoes_referencia() -> np.ndarray | None:
    """Pontuações das imagens do val (results/probabilidades_val_v2.csv), ou None se o
    arquivo não existir. O arquivo não vai para o git, porque lista imagens do dataset."""
    caminho = RESULTADOS / "probabilidades_val_v2.csv"
    return pd.read_csv(caminho)["prob_v2"].to_numpy() if caminho.exists() else None


def faixa_e_posicao(pontuacao: float, limiar: float, referencia: np.ndarray | None):
    """
    Faixa de decisão e posição relativa. "Revisar" se a pontuação for igual ou maior que o
    limiar congelado; senão, "Sem sinal relevante". A posição é a porcentagem de imagens
    de referência com pontuação menor (None sem referência).
    """
    faixa = "Revisar" if pontuacao >= limiar else "Sem sinal relevante"
    posicao = None if referencia is None else 100 * float((referencia < pontuacao).mean())
    return faixa, posicao


def analisar_imagem(caminho) -> dict:
    """
    Analisa uma imagem (PNG ou JPG, em cor, RGBA ou cinza; tudo vai para tons de cinza com
    fundo branco, como nas demais técnicas). Devolve as 12 features na ordem do modelo, a
    pontuação, a faixa, a posição entre as imagens de referência e os dados das
    visualizações: imagem e mapa de ELA na largura de exibição, mapa do Ghost (se houver
    histórico JPEG), pares de copy-move com as caixas, as regiões automáticas de ELA e a
    explicação em texto. Caixas e pares ficam em coordenadas da imagem original.
    """
    modelo, config = carregar_congelado()
    valores = {**extract_features(caminho), **component_features(caminho)}
    features = {nome: valores[nome] for nome in config["features"]}
    pontuacao = float(modelo.predict_proba(np.array([list(features.values())]))[0, 1])
    faixa, posicao = faixa_e_posicao(pontuacao, config["limiar"], pontuacoes_referencia())

    cinza = load_gray(caminho)
    escala = min(1.0, LARGURA_EXIBICAO / cinza.shape[1])
    tamanho = (round(cinza.shape[1] * escala), round(cinza.shape[0] * escala))
    exibicao = cv2.resize(np.clip(cinza, 0, 255).astype(np.uint8), tamanho, interpolation=cv2.INTER_AREA)
    # ELA bruto (sem amplificação): alimenta as caixas; amplificado ×15 (como compute_ela) é o mapa de exibição.
    ela_bruto = compute_ela(caminho, quality=90, amplification=1, grayscale=True)[0]
    mapa_ela = cv2.resize(np.clip(ela_bruto.astype(np.float32) * 15, 0, 255).astype(np.uint8), tamanho,
                          interpolation=cv2.INTER_AREA)
    cfg = carregar_regioes_config()
    regioes = regioes_ela(ela_bruto, cfg["percentil"], cfg["max_caixas"], cfg["contraste_min"])

    tem_historico = bool(features["has_jpeg_history"])
    pares, caixas = blind_copy_move(cinza, LIMIAR_COPY_MOVE)
    return {
        "features": features, "pontuacao": pontuacao, "limiar": config["limiar"],
        "faixa": faixa, "posicao": posicao,
        "escala": escala, "cinza_exibicao": exibicao, "mapa_ela": mapa_ela,
        "ghost_historico": tem_historico,
        "ghost_q90_frac": features["ghost_q90_frac"] if tem_historico else None,
        "ghost_mapa": compute_jpeg_ghost(caminho) if tem_historico else None,
        "ghost_qualidades": DEFAULT_QUALITIES,
        "copy_move_pares": pares, "copy_move_caixas": caixas,
        "regioes": [{"caixa": caixa, "intensidade": intensidade} for caixa, intensidade in regioes],
        "explicacao": explicar(regioes, (cinza.shape[1], cinza.shape[0]), pares, tem_historico, faixa),
    }
