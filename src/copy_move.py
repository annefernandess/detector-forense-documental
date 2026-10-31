"""
Copy-move (cópia dentro do mesmo documento) por correlação normalizada, o filtro
casado visto em aula: a região suspeita é usada como máscara e comparada com cada
posição da imagem. Se ela foi copiada de outro ponto do documento, a correlação com
esse ponto fica próxima de 1.
Fundamentação teórica (correlação de Pearson, filtro casado, limitações):
report/fundamentacao_copy_move.md (slides 260-272 do material da disciplina).
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from dataset_loader import load_gray
from evaluation import generate_pseudo_region, get_text_mask


def _clip_bbox(bbox, shape) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x, y = max(0, x), max(0, y)
    return x, y, min(w, shape[1] - x), min(h, shape[0] - y)


def _overlaps(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def correlation_map(gray: np.ndarray, bbox) -> tuple[np.ndarray | None, tuple | None]:
    """
    Correlação normalizada (coeficiente de Pearson em cada posição) entre a região
    `bbox` e a imagem. Posições que sobrepõem a própria região e janelas lisas
    (correlação indefinida) valem -1. Retorna (mapa, bbox recortada) ou (None, None)
    se a região for lisa.
    """
    x, y, w, h = _clip_bbox(bbox, gray.shape)
    template = gray[y:y + h, x:x + w]
    if template.size == 0 or template.std() == 0:
        return None, None
    corr = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    corr = np.nan_to_num(corr, nan=-1.0, posinf=-1.0, neginf=-1.0)
    corr[max(0, y - h + 1):y + h, max(0, x - w + 1):x + w] = -1.0
    return corr, (x, y, w, h)


def _peak(corr: np.ndarray, box) -> tuple[float, tuple[int, int, int, int]]:
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    return float(corr[py, px]), (int(px), int(py), box[2], box[3])


def correlation_peak(gray: np.ndarray, bbox) -> tuple[float | None, tuple | None]:
    """
    Filtro casado: maior correlação da região com outro ponto da imagem.
    Retorna (pico, bbox do ponto de maior correlação) ou (None, None) se a
    região for lisa.
    """
    corr, box = correlation_map(gray, bbox)
    return (None, None) if corr is None else _peak(corr, box)


def copy_move_region_score(image_path, regions: list) -> tuple[float | None, float | None]:
    """
    Score no formato de `evaluate_region_technique`: maior pico de correlação entre
    as regiões modificadas (áreas originais são ignoradas) e o pico de uma região de
    controle com texto, do tamanho da maior região, na mesma imagem.
    """
    gray = load_gray(image_path)
    targets = [r for r in regions if r.is_original_area is not True]
    peaks = [p for p, _ in (correlation_peak(gray, r.bbox) for r in targets) if p is not None]
    text_mask = get_text_mask(image_path)
    if not peaks or text_mask is None:
        return None, None
    _, _, w, h = _clip_bbox(max(targets, key=lambda r: r.bbox[2] * r.bbox[3]).bbox, gray.shape)
    control = generate_pseudo_region(gray.shape[1], gray.shape[0], w, h, regions, text_mask, random)
    if control is None:
        return max(peaks), None
    return max(peaks), correlation_peak(gray, control.bbox)[0]


def locate_peaks(sample) -> list[tuple]:
    """
    Para cada região modificada, indica onde cai o pico de correlação: "origem"
    (área original anotada), "outra região modificada", "fora das anotações" ou
    "sem origem anotada". Retorna (região, pico, destino) por região.
    """
    gray = load_gray(sample.image_path)
    origins = [r.bbox for r in sample.regions if r.is_original_area is True]
    located = []
    for r in sample.regions:
        if r.is_original_area is True:
            continue
        peak, loc = correlation_peak(gray, r.bbox)
        if peak is None:
            continue
        others = [o.bbox for o in sample.regions if o is not r and o.is_original_area is not True]
        if not origins:
            dest = "sem origem anotada"
        elif any(_overlaps(loc, o) for o in origins):
            dest = "origem"
        elif any(_overlaps(loc, o) for o in others):
            dest = "outra região modificada"
        else:
            dest = "fora das anotações"
        located.append((r, peak, dest))
    return located


def visualize_copy_move(image_path, bbox, title: str | None = None, axes=None):
    """
    Plota, em 1 linha x 3 colunas:
        [0] imagem com a região (vermelho) e o ponto de maior correlação (verde)
        [1] região ampliada
        [2] trecho de maior correlação ampliado
    Se `axes` for None, cria uma figura nova. Retorna a figura.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1, 1]})
    else:
        fig = axes[0].figure

    gray = load_gray(image_path)
    corr, box = correlation_map(gray, bbox)
    peak, loc = _peak(corr, box)

    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    for (x, y, w, h), color in ((box, "red"), (loc, "lime")):
        axes[0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2))
    axes[0].set_title(title or "Região (vermelho) e maior correlação (verde)", fontsize=9)

    for ax, (x, y, w, h), name in ((axes[1], box, "região"),
                                   (axes[2], loc, f"maior correlação (r = {peak:.3f})")):
        ax.imshow(gray[y:y + h, x:x + w], cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=9)

    for ax in axes:
        ax.axis("off")
    return fig
