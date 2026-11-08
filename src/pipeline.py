"""
Análise de uma imagem avulsa para a interface. Carrega o classificador candidato congelado
(results/modelo_candidato.joblib, com verificação do hash), calcula as 13 features na ordem de
results/configuracao_candidato.json (as 12 do v2 pela mesma extração do lote, mais `copy_move_par`, da técnica 1 de
copy_move_residuo com a configuração congelada) e devolve a faixa de decisão, a posição entre as previsões de
referência e os dados das visualizações. Não treina, não recalibra, não lê o test e não usa anotações.
O modelo v2 (`carregar_congelado`) continua disponível para o notebook 05, que o avalia.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from copy_move import candidate_boxes
from copy_move_residuo import candidatos, filtrar_pares
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
LIMIAR_COPY_MOVE = 0.99  # busca cega do v2: usado por avaliacao_sintetica (notebook 05); a interface usa a técnica 1


def verificar_hash(conteudo: bytes, esperado: str, nome: str) -> None:
    """Levanta RuntimeError se o SHA-256 do conteúdo não for o registrado."""
    if hashlib.sha256(conteudo).hexdigest() != esperado:
        raise RuntimeError(f"o hash de {nome} não confere com o registrado; parando")


@functools.lru_cache(maxsize=1)
def carregar_congelado():
    """(modelo, configuração) do v2, lidos de results/modelo_v2.joblib e configuracao_congelada.json."""
    config = json.loads((RESULTADOS / "configuracao_congelada.json").read_text(encoding="utf-8"))
    return joblib.load(RESULTADOS / "modelo_v2.joblib"), config


@functools.lru_cache(maxsize=1)
def carregar_candidato():
    """(modelo, configuração) do candidato, de results/modelo_candidato.joblib e configuracao_candidato.json.
    Confere o hash do modelo antes de usá-lo; se não bater, levanta RuntimeError."""
    config = json.loads((RESULTADOS / "configuracao_candidato.json").read_text(encoding="utf-8"))
    conteudo = (RESULTADOS / "modelo_candidato.joblib").read_bytes()
    verificar_hash(conteudo, config["modelo"]["sha256"], "results/modelo_candidato.joblib")
    return joblib.load(io.BytesIO(conteudo)), config


@functools.lru_cache(maxsize=1)
def carregar_regioes_config() -> dict:
    """Parâmetros das caixas automáticas e medidas da calibração, de results/regioes_config.json
    (calibrado no val)."""
    return json.loads((RESULTADOS / "regioes_config.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def metricas_candidato() -> dict:
    """Sensibilidade e especificidade do candidato no limiar congelado, nas previsões fora da amostra da validação
    cruzada de train + val (results/configuracao_candidato.json). Não são medidas do test."""
    limiar = carregar_candidato()[1]["limiar"]
    return {"sensibilidade": limiar["sensibilidade"], "especificidade": limiar["especificidade"]}


@functools.lru_cache(maxsize=1)
def pontuacoes_referencia() -> np.ndarray | None:
    """Previsões fora da amostra da validação cruzada do candidato (results/referencia_percentil_candidato.json,
    train + val), com o hash conferido; None se o arquivo não existir."""
    caminho = RESULTADOS / "referencia_percentil_candidato.json"
    if not caminho.exists():
        return None
    referencia = json.loads(caminho.read_text(encoding="utf-8"))
    previsoes = np.array(referencia["previsoes"], dtype=np.float64)
    verificar_hash(previsoes.tobytes(), referencia["sha256_previsoes"], "results/referencia_percentil_candidato.json")
    return previsoes


def faixa_e_posicao(pontuacao: float, limiar: float, referencia: np.ndarray | None):
    """
    Faixa de decisão e posição relativa. "Revisar" se a pontuação for igual ou maior que o
    limiar congelado; senão, "Sem sinal relevante". A posição é a porcentagem de previsões
    de referência com pontuação menor (None sem referência).
    """
    faixa = "Revisar" if pontuacao >= limiar else "Sem sinal relevante"
    posicao = None if referencia is None else 100 * float((referencia < pontuacao).mean())
    return faixa, posicao


def analisar_imagem(caminho) -> dict:
    """
    Analisa uma imagem (PNG ou JPG, em cor, RGBA ou cinza; tudo vai para tons de cinza com
    fundo branco, como nas demais técnicas). Devolve as 13 features na ordem do modelo, a
    pontuação, a faixa, a posição entre as previsões de referência e os dados das
    visualizações: imagem e mapa de ELA na largura de exibição, mapa do Ghost (se houver
    histórico JPEG), pares de copy-move aprovados pela técnica 1 com as caixas, as regiões
    automáticas de ELA e a explicação em texto. Caixas e pares ficam em coordenadas da imagem
    original.
    """
    modelo, config = carregar_candidato()
    tecnica = config["configuracao_tecnica_1"]
    cinza = load_gray(caminho)
    # a mesma chamada de copy_move_feature.feature_copy_move, mas guardando os pares para a visualização
    pares = filtrar_pares(candidatos(cinza, ks=(tecnica["k"],)), tecnica["piso_intensidade"], tecnica["k"],
                          tecnica["limiar_residuo"])
    pares = [p[:3] for p in pares]
    valores = {**extract_features(caminho), **component_features(caminho), "copy_move_par": int(len(pares) > 0)}
    features = {nome: valores[nome] for nome in config["features"]}
    pontuacao = float(modelo.predict_proba(np.array([list(features.values())]))[0, 1])
    limiar = config["limiar"]["valor"]
    faixa, posicao = faixa_e_posicao(pontuacao, limiar, pontuacoes_referencia())

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
    return {
        "features": features, "pontuacao": pontuacao, "limiar": limiar,
        "faixa": faixa, "posicao": posicao,
        "escala": escala, "cinza_exibicao": exibicao, "mapa_ela": mapa_ela,
        "ghost_historico": tem_historico,
        "ghost_q90_frac": features["ghost_q90_frac"] if tem_historico else None,
        "ghost_mapa": compute_jpeg_ghost(caminho) if tem_historico else None,
        "ghost_qualidades": DEFAULT_QUALITIES,
        "copy_move_pares": pares, "copy_move_caixas": candidate_boxes(pares),
        "regioes": [{"caixa": caixa, "intensidade": intensidade} for caixa, intensidade in regioes],
        "explicacao": explicar(regioes, (cinza.shape[1], cinza.shape[0]), pares, tem_historico, faixa),
    }
