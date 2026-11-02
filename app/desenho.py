"""Desenho das regiões automáticas de ELA na interface (caixas numeradas, sem anotações)."""

from __future__ import annotations

import cv2
import numpy as np

MAGENTA = (220, 20, 200)


def desenhar_regioes(rgb: np.ndarray, regioes: list, escala: float) -> np.ndarray:
    """Caixas numeradas por ordem de intensidade de ELA. `regioes` vem de `analisar_imagem`
    (caixas em coordenadas da imagem original); `escala` leva à largura de exibição."""
    saida = rgb.copy()
    for i, r in enumerate(regioes, 1):
        x, y, w, h = (round(v * escala) for v in r["caixa"])
        cv2.rectangle(saida, (x, y), (x + max(w, 8), y + max(h, 8)), MAGENTA, 3)
        cv2.putText(saida, str(i), (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, MAGENTA, 2, cv2.LINE_AA)
    return saida
