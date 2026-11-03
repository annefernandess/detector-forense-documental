"""
Ajuste dos parâmetros das técnicas clássicas (ELA, JPEG Ghost, busca cega de copy-move) pela
localização da caixa colada num conjunto sintético de desenvolvimento. A grade e o critério estão
congelados em results/grade_ajuste.json. O classificador não é tocado. A medida de cada
configuração é o acerto na caixa registrada contra o acaso (evaluate_region_technique, seed 42).
"""

from __future__ import annotations

import functools
import random
from multiprocessing import get_context

import cv2
import numpy as np

from copy_move import blind_copy_move_for_path, candidate_boxes, filter_pairs, region_covered
from dataset_loader import read_gray_cv2
from ela import compute_ela
from evaluation import evaluate_region_technique, generate_pseudo_region, get_text_mask
from jpeg_ghost import make_ghost_score_fn

GRADE = {
    "ela": {"qualidade": (80, 90), "percentil": (95, 99)},
    "ghost": {"block_size": (8, 16, 32)},
    "copy_move": {"limiar": (0.95, 0.979, 0.99), "cobertura": (0.3, 0.5)},
}
# ELA: padrão = média do erro na região (sem percentil), q=90; é a medida do notebook 01.
PADROES = {"ela": {"qualidade": 90, "percentil": None}, "ghost": {"block_size": 16},
           "copy_move": {"limiar": 0.99, "cobertura": 0.5}}
GANHO_MIN = 0.05       # ganho mínimo, em diferença (acerto real - acaso) sobre a padrão
PISO_PARES = 0.95      # pares da busca cega guardados a partir deste r; os limiares maiores filtram depois
_AMOSTRAS: list = []   # preenchidos antes do fork, para os processos herdarem sem copiar
_PARES: dict = {}


def configuracoes() -> list:
    """(técnica, parâmetros) da padrão de cada técnica e de toda a grade, sem repetir a padrão."""
    lista = []
    for tecnica, padrao in PADROES.items():
        nomes = list(GRADE[tecnica])
        grade = [{}]
        for nome in nomes:
            grade = [{**g, nome: v} for g in grade for v in GRADE[tecnica][nome]]
        lista.append((tecnica, padrao))
        lista += [(tecnica, g) for g in grade if g != padrao]
    return lista


@functools.lru_cache(maxsize=2)
def _mapa_ela(caminho: str, qualidade: int):
    return compute_ela(caminho, quality=qualidade, amplification=1, grayscale=True)[0]


def ela_score_fn(qualidade: int, percentil):
    """score_fn da ELA: média (percentil None) ou percentil do erro na caixa e num controle com texto."""
    def score(caminho, regioes):
        mapa = _mapa_ela(str(caminho), qualidade)
        alvos = [r for r in regioes if r.is_original_area is not True]
        texto = get_text_mask(str(caminho))
        if not alvos or texto is None:
            return None, None
        funcao = np.mean if percentil is None else functools.partial(np.percentile, q=percentil)
        x, y, w, h = max(alvos, key=lambda r: r.bbox[2] * r.bbox[3]).bbox
        controle = generate_pseudo_region(mapa.shape[1], mapa.shape[0], w, h, regioes, texto, random)
        if controle is None:
            return None, None
        cx, cy, cw, ch = controle.bbox
        dentro = [mapa[b[1]:b[1] + b[3], b[0]:b[0] + b[2]] for b in (r.bbox for r in alvos)]
        return float(funcao(np.concatenate([d.ravel() for d in dentro]))), float(funcao(mapa[cy:cy + ch, cx:cx + cw]))
    return score


def busca_cega_score_fn(limiar: float, cobertura: float, pares: dict):
    """score_fn da busca cega: 1.0 se uma caixa detectada (r >= limiar) cobre >= `cobertura` de alguma região."""
    def score(caminho, regioes):
        caixas = candidate_boxes(filter_pairs(pares[str(caminho)], limiar))
        alvos = [r for r in regioes if r.is_original_area is not True]
        return float(any(region_covered(r.bbox, caixas, cobertura) for r in alvos)), 0.0
    return score


def _pares(caminho):
    cv2.setNumThreads(1)
    return str(caminho), blind_copy_move_for_path(caminho, PISO_PARES)[0]


def calcular_pares(amostras: list, processos: int = 8) -> dict:
    """Pares da busca cega (r >= PISO_PARES) de cada imagem, calculados em paralelo uma só vez."""
    with get_context("fork").Pool(processos) as pool:
        return dict(pool.map(_pares, [s.image_path for s in amostras]))


def avaliar(tecnica: str, params: dict, amostras: list, pares: dict | None = None) -> dict:
    """Linha de resultado de uma configuração: acerto real e acaso, com Wilson, e exclusões."""
    cv2.setNumThreads(1)
    if tecnica == "ela":
        r = evaluate_region_technique(amostras, ela_score_fn(params["qualidade"], params["percentil"]),
                                      lambda sin, sout: sin > sout)
    elif tecnica == "ghost":
        r = evaluate_region_technique(amostras, make_ghost_score_fn(params["block_size"]),
                                      lambda sin, sout: sin != sout)
    else:
        r = evaluate_region_technique(amostras, busca_cega_score_fn(params["limiar"], params["cobertura"], pares),
                                      lambda sin, sout: sin >= 1.0)
    return {"tecnica": tecnica, **params, "n": r["n_samples"], "acertos": r["real_hits"],
            "taxa_real": r["real_hit_rate"], "real_ic_min": r["real_wilson"][0], "real_ic_max": r["real_wilson"][1],
            "taxa_acaso": r["chance_hit_rate"], "acaso_ic_min": r["chance_wilson"][0],
            "acaso_ic_max": r["chance_wilson"][1], "diferenca": r["real_hit_rate"] - r["chance_hit_rate"],
            "exclusoes": str(r["exclusions"])}


def _avaliar_config(config):
    return avaliar(config[0], config[1], _AMOSTRAS, _PARES)


def avaliar_grade(amostras: list, processos: int = 8) -> list:
    """Avalia toda a grade (uma configuração por processo). Calcula os pares da busca cega antes."""
    global _AMOSTRAS, _PARES
    _AMOSTRAS, _PARES = amostras, calcular_pares(amostras, processos)
    with get_context("fork").Pool(processos) as pool:
        return pool.map(_avaliar_config, configuracoes())


def escolher(linhas: list, padrao: dict) -> tuple:
    """
    Configuração escolhida entre as linhas de uma técnica: a de maior diferença para o acaso que
    ganha >= GANHO_MIN da padrão em diferença E cujo IC de Wilson do acerto real não toca o IC do
    acaso (real_ic_min > acaso_ic_max). Sem nenhuma que cumpra, a padrão. Devolve (linha, trocou).
    """
    chaves = [k for k in padrao]
    base = next(l for l in linhas if all(l[k] == padrao[k] for k in chaves))
    validas = [l for l in linhas if l is not base and l["diferenca"] - base["diferenca"] >= GANHO_MIN
               and l["real_ic_min"] > l["acaso_ic_max"]]
    return (max(validas, key=lambda l: l["diferenca"]), True) if validas else (base, False)
