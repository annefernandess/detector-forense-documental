"""
Gerador de copy-move sintético para avaliação externa. Dada uma imagem em tons de cinza,
escolhe um componente conexo da máscara de Otsu (uma "caixa de texto") e cola o retângulo da
sua caixa sobre outro ponto de texto da mesma imagem, sem sobreposição com a origem. Os pixels
colados são copiados sem nenhuma transformação (sem rotação, escala, ruído ou recompressão).

Os parâmetros abaixo estão congelados em results/gerador_congelado.json, com o hash deste
arquivo. Não alterar depois de iniciar a avaliação.
"""

from __future__ import annotations

import cv2
import numpy as np

SEED = 42
LADO_MIN_FRACAO = 0.010   # lado mínimo da caixa, como fração da largura da imagem
LADO_MAX_FRACAO = 0.060   # lado máximo da caixa, como fração da largura da imagem
AREA_TINTA_MIN = 20       # pixels de texto mínimos no componente de origem
TINTA_MIN_ORIGEM = 0.15   # fração mínima de texto dentro da caixa de origem
TINTA_MIN_DESTINO = 0.10  # fração mínima de texto dentro da caixa de destino (antes da colagem)
AREA_MIN_DESTINO = 15     # pixels de texto mínimos no componente que define o ponto de destino
TENTATIVAS_DESTINO = 200


def mascara_texto(gray: np.ndarray) -> np.ndarray:
    """Máscara booleana de Otsu do texto (escuro sobre claro), como em features._text_mask."""
    _, mask = cv2.threshold(gray.astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(mask) > mask.size / 2:
        mask = cv2.bitwise_not(mask)
    return mask > 0


def fracao_texto(mask: np.ndarray, caixa: tuple) -> float:
    """Fração de pixels de texto dentro da caixa (x, y, w, h)."""
    x, y, w, h = caixa
    return float(mask[y:y + h, x:x + w].mean())


def _sobrepoe(a: tuple, b: tuple) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def colar_caixa_texto(gray: np.ndarray, rng: np.random.Generator):
    """
    Copia a caixa de um componente de texto para o ponto de outro componente da mesma imagem.
    Retorna (imagem editada, caixa colada (x, y, w, h)). A caixa devolvida é a do destino.
    Levanta ValueError se não houver par origem/destino válido.
    """
    altura, largura = gray.shape
    mask = mascara_texto(gray)
    n, _, stats, centros = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    lado_min, lado_max = LADO_MIN_FRACAO * largura, LADO_MAX_FRACAO * largura

    origens = [i for i in range(1, n)
               if lado_min <= stats[i, 2] <= lado_max and lado_min <= stats[i, 3] <= lado_max
               and stats[i, 4] >= AREA_TINTA_MIN
               and fracao_texto(mask, tuple(int(v) for v in stats[i, :4])) >= TINTA_MIN_ORIGEM]
    destinos = [i for i in range(1, n) if stats[i, 4] >= AREA_MIN_DESTINO]
    for i in rng.permutation(origens):
        origem = tuple(int(v) for v in stats[i, :4])
        w, h = origem[2], origem[3]
        for j in rng.permutation(destinos)[:TENTATIVAS_DESTINO]:
            if j == i:
                continue
            x = int(round(centros[j, 0] - w / 2))
            y = int(round(centros[j, 1] - h / 2))
            caixa = (x, y, w, h)
            if x < 0 or y < 0 or x + w > largura or y + h > altura or _sobrepoe(caixa, origem):
                continue
            if fracao_texto(mask, caixa) < TINTA_MIN_DESTINO:
                continue
            editada = gray.copy()
            editada[y:y + h, x:x + w] = gray[origem[1]:origem[1] + h, origem[0]:origem[0] + w]
            return editada, caixa
    raise ValueError("nenhum par origem/destino válido nesta imagem")
