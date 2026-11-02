"""
Caixas automáticas de região suspeita a partir do mapa de ELA (sem amplificação, valores em
níveis de cinza). Nenhuma anotação é usada. O mapa vai para largura de 1000 px (a mesma escala da
busca cega de copy-move), é suavizado com filtro gaussiano (slides 201-207) e limiarizado por um
percentil da própria imagem; as caixas são os componentes conexos que sobram. Não usa
morfologia (erosão e dilatação não fazem parte da disciplina).
"""

from __future__ import annotations

import cv2
import numpy as np

LARGURA = 1000
SIGMA = 2.0          # suavização gaussiana, em px na escala de 1000 px: apaga pontos isolados
AREA_MIN = 200       # caixa menor que um caractere (cerca de 15 x 27 px nesta escala) é descartada
FRACAO_MAX = 0.20    # caixa com mais de 20% da área da imagem é descartada


def mapa_para_1000(mapa_ela: np.ndarray) -> np.ndarray:
    """Mapa de ELA em float32 na largura de 1000 px (INTER_AREA ao reduzir, INTER_LINEAR ao ampliar)."""
    escala = LARGURA / mapa_ela.shape[1]
    return cv2.resize(mapa_ela.astype(np.float32), (LARGURA, round(mapa_ela.shape[0] * escala)),
                      interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR)


def regioes_ela(mapa_ela: np.ndarray, percentil: float, max_caixas: int,
                contraste_min: float = 0.0) -> list:
    """
    Regiões de maior diferença de compressão. O limiar é o maior entre o percentil `percentil`
    do mapa suavizado e a mediana mais 1 nível de cinza, para não depender do contraste global
    nem segmentar ruído em páginas quase uniformes. Componentes com `contraste_min` níveis de
    cinza a menos que a mediana são descartados.

    Retorna até `max_caixas` itens ((x, y, w, h), intensidade) em coordenadas do mapa de
    entrada, ordenados pela intensidade média de ELA (maior primeiro). A intensidade é medida
    no mapa de 1000 px, em níveis de cinza.
    """
    escala = LARGURA / mapa_ela.shape[1]
    suave = cv2.GaussianBlur(mapa_para_1000(mapa_ela), (0, 0), SIGMA)
    mediana = float(np.median(suave))
    limiar = max(float(np.percentile(suave, percentil)), mediana + 1.0)
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats((suave >= limiar).astype(np.uint8), connectivity=8)

    candidatos = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w * h < AREA_MIN or w * h > FRACAO_MAX * suave.size:
            continue
        intensidade = float(suave[rotulos == i].mean())
        if intensidade - mediana < contraste_min:
            continue
        candidatos.append(((round(x / escala), round(y / escala), round(w / escala), round(h / escala)), intensidade))
    return sorted(candidatos, key=lambda c: -c[1])[:max_caixas]
