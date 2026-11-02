"""
Features v3 para o classificador (notebooks/arquivo/06_alinhamento_sem_ganho.ipynb,
tentativa sem ganho): tamanho do texto, z-score máximo
normalizado e alinhamento de linha de base. Nenhuma feature usa as anotações.
"""

from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dataset_loader import load_gray
from features import _init_worker, _text_mask
from features_v2 import _components, _robust_z, component_values

V3_NAMES = ["n_components", "ela_comp_top3_z", "noise_comp_top3_z", "edge_comp_top3_z",
            "intensity_comp_top3_z", "align_max_z", "align_frac_out"]
TOP3_NAMES = V3_NAMES[1:5]
ALIGN_NAMES = ["align_max_z", "align_frac_out"]


def _top3_abs_z(values: np.ndarray) -> float:
    """Média dos 3 maiores |z| robustos: menos sensível ao número de componentes que o máximo."""
    z = np.sort(np.abs(_robust_z(values)[0]))
    return float(z[-3:].mean())


def _resize_width(gray: np.ndarray, width: int = 1000) -> np.ndarray:
    scale = width / gray.shape[1]
    return cv2.resize(gray, (width, round(gray.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def _text_lines(mask: np.ndarray, min_ink: float = 0.005) -> list:
    """Linhas de texto pelo perfil de projeção horizontal: faixas contínuas de linhas da
    imagem com pelo menos `min_ink` da largura em pixels de texto. Retorna (início, fim)."""
    ink = mask.sum(axis=1) >= min_ink * mask.shape[1]
    lines, start = [], None
    for r, on in enumerate(np.append(ink, False)):
        if on and start is None:
            start = r
        elif not on and start is not None:
            lines.append((start, r - 1))
            start = None
    return lines


def baseline_deviations(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Desvio de cada componente em relação à linha de base da própria linha de texto.

    A imagem é reescalada para 1000 px de largura, para que um pixel tenha o mesmo
    significado em todas as imagens. As linhas vêm do perfil de projeção horizontal da
    máscara de Otsu (histograma das linhas da imagem: slides 232-236). Em cada linha, a
    linha de base é uma reta ajustada às bases (y + h) dos componentes, com uma rodada
    de remoção de discrepantes (|resíduo| > 2 escalas robustas), o que tolera inclinação
    da digitalização e letras descendentes.

    Retorna (desvio em pixels, desvio dividido pela altura mediana dos caracteres da linha).

    Ressalva: se o editor de imagem alinhar bem o texto inserido, o desvio fica igual ao
    do texto original e o sinal some.
    """
    small = _resize_width(gray)
    lines = _text_lines(_text_mask(small))
    _, boxes, _ = _components(small)
    if len(boxes) == 0:
        return np.empty(0), np.empty(0)
    x, y, w, h = boxes.T.astype(float)
    bottom, xc, yc = y + h, x + w / 2, y + h / 2
    dev_px, dev_norm = [], []
    for r0, r1 in lines:
        idx = np.nonzero((yc >= r0) & (yc <= r1))[0]
        if len(idx) < 3:
            continue
        a, b = np.polyfit(xc[idx], bottom[idx], 1)
        res = bottom[idx] - (a * xc[idx] + b)
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        inliers = idx[np.abs(res) <= 2 * mad] if mad > 0 else idx
        if len(inliers) >= 3:
            a, b = np.polyfit(xc[inliers], bottom[inliers], 1)
        d = bottom[idx] - (a * xc[idx] + b)
        dev_px.extend(d)
        dev_norm.extend(d / np.median(h[idx]))
    return np.array(dev_px), np.array(dev_norm)


def v3_features(image_path, min_components: int = 5) -> dict:
    """
    Sete features por imagem:
      n_components: número de componentes de texto válidos (escala original);
      *_top3_z: média dos 3 maiores |z| robustos por componente, para ELA (slides 127, 133,
          376-379, 432-440), ruído da mediana 3x3 (201, 208-216), Sobel (225-230) e
          intensidade (232-236);
      align_max_z: maior |z| robusto do desvio normalizado em relação à linha de base;
      align_frac_out: fração de componentes com |desvio| > 1 pixel na escala de 1000 px.
    Com menos de `min_components` componentes, as features de z valem 0.
    """
    out = dict.fromkeys(V3_NAMES, 0.0)
    values = component_values(image_path)
    n = 0 if values is None else len(values["ela"])
    out["n_components"] = float(n)
    if n >= min_components:
        for name, key in zip(TOP3_NAMES, ("ela", "noise", "edge", "intensity")):
            out[name] = _top3_abs_z(values[key])
    dev_px, dev_norm = baseline_deviations(load_gray(image_path))
    if len(dev_norm) >= min_components:
        out["align_max_z"] = float(np.abs(_robust_z(dev_norm)[0]).max())
        out["align_frac_out"] = float((np.abs(dev_px) > 1.0).mean())
    return out


def load_or_extract_v3(samples: list, cache_path, processes: int = 10) -> pd.DataFrame:
    """Tabela com nome da imagem e as features v3. Lê de `cache_path` (CSV) se existir;
    senão calcula em paralelo e salva."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_csv(cache_path)
    with get_context("fork").Pool(processes, initializer=_init_worker) as pool:
        rows = pool.map(v3_features, [s.image_path for s in samples])
    table = pd.DataFrame(rows, columns=V3_NAMES)
    table.insert(0, "image", [s.image_path.name for s in samples])
    table.to_csv(cache_path, index=False)
    return table
