"""
Busca cega de copy-move com verificação do resíduo de ruído. Os pares candidatos vêm da busca
cega atual (`blind_copy_move`: largura de 1000 px, componentes de Otsu, caixas de mesmo tamanho,
correlação de intensidade >= piso), já com as caixas na resolução original. Para cada par, o
resíduo (imagem menos a mediana k x k, em tons de cinza, na resolução original) das duas caixas
é correlacionado. Um caractere impresso duas vezes tem ruído de digitalização diferente em
cada ocorrência; uma cópia digital carrega o mesmo ruído. O par só é mantido se a correlação do
resíduo atingir o limiar. Nada aqui altera `copy_move.py`.

Grade e critérios congelados em results/grade_residuo.json (hash deste arquivo).
"""

from __future__ import annotations

import itertools
from multiprocessing import get_context

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from copy_move import blind_copy_move, candidate_boxes, correlation_peak, filter_pairs, region_covered
from dataset_loader import load_gray

KS = (3, 5)           # tamanhos do filtro da mediana (cv2.medianBlur em float32 aceita só 3 e 5)
PISO_PARES = 0.95     # menor piso de intensidade da grade; os pisos maiores filtram depois
DESLOCAMENTO = 2      # px, em x e y, para compensar o arredondamento do mapeamento 1000 px -> original
MIN_SIDE = 6          # fixo (padrão da busca cega)
COBERTURA_MIN = 0.5   # fixa (padrão de region_covered)


def residuo(gray: np.ndarray, k: int) -> np.ndarray:
    """Imagem menos a mediana k x k (float32), na resolução da imagem."""
    gray = gray.astype(np.float32)
    return gray - cv2.medianBlur(gray, k)


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Correlação de Pearson de dois trechos; None se algum for liso (correlação indefinida)."""
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    denominador = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denominador < 1e-9 else float(a @ b / denominador)


def _caixa_na_imagem(caixa, shape) -> tuple[int, int, int, int] | None:
    """Caixa (x, y, w, h) com a origem deslocada para caber na imagem (o arredondamento do
    mapeamento pode passar 1 px da borda); None se a caixa for maior que a imagem."""
    x, y, w, h = caixa
    if w > shape[1] or h > shape[0]:
        return None
    return min(max(x, 0), shape[1] - w), min(max(y, 0), shape[0] - h), w, h


def correlacao_residuo(res: np.ndarray, caixa_a, caixa_b, deslocamento: int = DESLOCAMENTO) -> float | None:
    """
    Maior correlação de Pearson entre o resíduo de `caixa_a` e o de `caixa_b` deslocada de até
    `deslocamento` px em cada eixo. As caixas têm o mesmo tamanho. None se nenhuma posição der
    correlação definida.
    """
    a = _caixa_na_imagem(caixa_a, res.shape)
    if a is None or caixa_b[2:] != caixa_a[2:]:
        return None
    ax, ay, w, h = a
    trecho_a = res[ay:ay + h, ax:ax + w]
    melhor = None
    for dy, dx in itertools.product(range(-deslocamento, deslocamento + 1), repeat=2):
        b = _caixa_na_imagem((caixa_b[0] + dx, caixa_b[1] + dy, w, h), res.shape)
        r = _pearson(trecho_a, res[b[1]:b[1] + h, b[0]:b[0] + w])
        if r is not None and (melhor is None or r > melhor):
            melhor = r
    return melhor


def candidatos(gray: np.ndarray, ks=KS) -> list:
    """Pares candidatos (busca cega com piso PISO_PARES) e a correlação do resíduo de cada um, por k:
    lista de (origem, destino, correlação de intensidade, {k: correlação do resíduo ou None})."""
    pares, _ = blind_copy_move(gray, PISO_PARES, min_side=MIN_SIDE)
    residuos = {k: residuo(gray, k) for k in ks}
    return [(src, dst, r, {k: correlacao_residuo(residuos[k], src, dst) for k in ks}) for src, dst, r in pares]


def filtrar_pares(cands: list, piso: float, k: int | None = None, limiar: float | None = None) -> list:
    """
    Pares com intensidade >= `piso` e, se `k` for dado, resíduo >= `limiar` (par sem resíduo
    definido é descartado). Remove redundância como `filter_pairs`. Retorna (origem, destino,
    intensidade, resíduo ou None).
    """
    residuo_de = {(s, d): res[k] if k is not None else None for s, d, _, res in cands}
    mantidos = [(s, d, r) for s, d, r, res in cands
                if r >= piso and (k is None or (res[k] is not None and res[k] >= limiar))]
    return [(s, d, r, residuo_de[(s, d)]) for s, d, r in filter_pairs(mantidos, piso)]


def n_sem_residuo(cands: list, k: int) -> int:
    """Quantos pares candidatos não têm correlação de resíduo definida (trecho liso)."""
    return sum(res[k] is None for _, _, _, res in cands)


def cobertura(cands_por_imagem: dict, regioes: dict, piso: float, k=None, limiar=None) -> tuple[int, int]:
    """(regiões cobertas, total). `regioes`: imagem -> lista de caixas; uma região é coberta se
    alguma caixa candidata cobre >= COBERTURA_MIN dela (como no notebook 03)."""
    cobertas = total = 0
    for nome, caixas in regioes.items():
        boxes = candidate_boxes([p[:3] for p in filtrar_pares(cands_por_imagem[nome], piso, k, limiar)])
        for caixa in caixas:
            total += 1
            cobertas += region_covered(caixa, boxes, COBERTURA_MIN)
    return cobertas, total


def avaliar_config(cands_cpi: dict, regioes_cpi: dict, cands_aut: dict, piso: float, k=None, limiar=None) -> dict:
    """Linha da grade: imagens com CPI com algum par, autênticas com algum par, J, cobertura das
    regiões CPI e pares por autêntica. `k=None` é a busca atual (só intensidade)."""
    com_par = lambda cands: [len(filtrar_pares(c, piso, k, limiar)) for c in cands.values()]
    cpi, aut = com_par(cands_cpi), com_par(cands_aut)
    cobertas, total = cobertura(cands_cpi, regioes_cpi, piso, k, limiar)
    tpr, fpr = float(np.mean([n > 0 for n in cpi])), float(np.mean([n > 0 for n in aut]))
    return {"piso": piso, "k": k, "limiar_residuo": limiar, "taxa_cpi": tpr, "taxa_autenticas": fpr,
            "J": tpr - fpr, "cobertura": cobertas / total, "regioes_cobertas": cobertas, "regioes_total": total,
            "pares_por_autentica": float(np.mean(aut)),
            "pares_por_autentica_com_par": float(np.mean([n for n in aut if n > 0])) if any(aut) else 0.0}


def escolher(linhas: list) -> dict:
    """Maior J; desempate pela maior cobertura de regiões CPI (critério congelado)."""
    return max(linhas, key=lambda l: (l["J"], l["cobertura"]))


def separacao_auc(positivos: list, negativos: list) -> float:
    """AUC de separar os dois grupos pelo valor (positivos esperados com valor maior)."""
    y = [1] * len(positivos) + [0] * len(negativos)
    return float(roc_auc_score(y, list(positivos) + list(negativos)))


def _candidatos_caminho(caminho):
    cv2.setNumThreads(1)
    return str(caminho), candidatos(load_gray(caminho))


def candidatos_lote(caminhos: list, processos: int = 8) -> dict:
    """`candidatos` de cada imagem (caminho -> lista), em paralelo."""
    with get_context("fork").Pool(processos) as pool:
        return dict(pool.map(_candidatos_caminho, caminhos))


def _regioes_cpi_caminho(argumento):
    """Correlação do resíduo entre cada região CPI e o ponto do pico de correlação (como no notebook 03)."""
    cv2.setNumThreads(1)
    caminho, caixas = argumento
    gray = load_gray(caminho)
    residuos = {k: residuo(gray, k) for k in KS}
    linhas = []
    for caixa in caixas:
        pico, loc = correlation_peak(gray, caixa)
        if pico is None:
            continue
        origem = (max(0, caixa[0]), max(0, caixa[1]), loc[2], loc[3])  # mesmo recorte de correlation_map
        linhas.append({"imagem": str(caminho), "pico": pico,
                       **{f"residuo_k{k}": correlacao_residuo(residuos[k], origem, loc) for k in KS}})
    return linhas


def correlacoes_regioes_cpi(regioes_por_caminho: dict, processos: int = 8) -> list:
    """Linhas {imagem, pico, residuo_k3, residuo_k5} das regiões CPI (caminho -> lista de caixas)."""
    with get_context("fork").Pool(processos) as pool:
        return [l for linhas in pool.map(_regioes_cpi_caminho, regioes_por_caminho.items()) for l in linhas]
