"""
Features por componente de texto para o classificador (notebook 04). Para cada
componente conexo da máscara de Otsu, mede-se um valor (erro da ELA, ruído, gradiente,
intensidade) e compara-se com os demais componentes da mesma página por um z-score
robusto. A ideia: uma região editada deve destoar do resto da própria página, e não de
um valor global. Nenhuma feature usa as anotações.
"""

from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dataset_loader import load_gray
from ela import compute_ela
from features import _init_worker, _text_mask

V2_NAMES = ["ela_comp_max_z", "ela_comp_frac_out", "noise_comp_max_z",
            "edge_comp_max_z", "intensity_comp_max_z"]


def _components(gray: np.ndarray):
    """
    Componentes conexos de texto (máscara de Otsu). Mantém área >= max(10, 20% da
    mediana das áreas), altura <= 5% da imagem e largura <= 20% da imagem, para tirar
    pontos de ruído, linhas de tabela e blocos. Retorna (rótulos, caixas, ids mantidos).
    """
    H, W = gray.shape
    _, labels, stats, _ = cv2.connectedComponentsWithStats(_text_mask(gray).astype(np.uint8), connectivity=8)
    stats = stats[1:]
    if len(stats) == 0:
        return labels, np.empty((0, 4), int), np.empty(0, int)
    area = stats[:, cv2.CC_STAT_AREA]
    keep = ((area >= max(10, 0.2 * np.median(area)))
            & (stats[:, cv2.CC_STAT_HEIGHT] <= 0.05 * H)
            & (stats[:, cv2.CC_STAT_WIDTH] <= 0.2 * W))
    ids = np.nonzero(keep)[0] + 1  # rótulo 0 é o fundo
    return labels, stats[keep, :4], ids


def _box_means(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Média de `image` em cada caixa (x, y, w, h), por imagem integral."""
    ii = cv2.integral(image.astype(np.float64))
    x, y, w, h = boxes.T
    total = ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x]
    return total / (w * h)


def _robust_z(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """z-score robusto: (v − mediana) / (1,4826 · MAD). Retorna (z, mediana, escala);
    se a escala for zero, z vale 0. MAD e z-score não aparecem nos slides."""
    med = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - med)))
    if scale < 1e-9:
        return np.zeros_like(values, dtype=float), med, 0.0
    return (values - med) / scale, med, scale


def component_values(image_path) -> dict | None:
    """
    Valores por componente de texto: erro médio da ELA (qualidade 90), desvio do resíduo
    da mediana 3x3 e magnitude média de Sobel na caixa, e intensidade média dos pixels do
    componente. Retorna dict de arrays (e as caixas), ou None se não houver componentes.
    """
    gray = load_gray(image_path)
    labels, boxes, ids = _components(gray)
    if len(ids) == 0:
        return None
    ela_map, _ = compute_ela(image_path, quality=90, amplification=1, grayscale=True)
    u8 = np.clip(gray, 0, 255).astype(np.uint8)
    residual = u8.astype(np.float64) - cv2.medianBlur(u8, 3).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    pixel_sum = np.bincount(labels.ravel(), weights=gray.ravel(), minlength=labels.max() + 1)
    pixel_count = np.bincount(labels.ravel(), minlength=labels.max() + 1)
    return {
        "boxes": boxes,
        "ela": _box_means(ela_map, boxes),
        "noise": np.sqrt(np.maximum(_box_means(residual ** 2, boxes) - _box_means(residual, boxes) ** 2, 0)),
        "edge": _box_means(np.hypot(gx, gy), boxes),
        "intensity": pixel_sum[ids] / pixel_count[ids],
    }


def component_features(image_path, min_components: int = 5) -> dict:
    """
    Cinco features por imagem, a partir dos componentes de texto:
      ela_comp_max_z: maior |z| do erro médio da ELA (qualidade 90) na caixa do componente
          (DCT e quantização: slides 127, 133, 376-379, 432-440);
      ela_comp_frac_out: fração de componentes com erro da ELA > mediana + 3 escalas robustas;
      noise_comp_max_z: maior |z| do desvio do resíduo (imagem − mediana 3x3) na caixa
          (filtro da mediana: slides 201, 208-216);
      edge_comp_max_z: maior |z| da magnitude média do gradiente de Sobel na caixa
          (slides 225-230);
      intensity_comp_max_z: maior |z| da intensidade média dos pixels do componente
          (histograma: slides 232-236).
    Com menos de `min_components` componentes, todas valem 0.
    """
    values = component_values(image_path)
    if values is None or len(values["ela"]) < min_components:
        return dict.fromkeys(V2_NAMES, 0.0)
    ela, noise, edge, intensity = values["ela"], values["noise"], values["edge"], values["intensity"]

    z_ela, med, scale = _robust_z(ela)
    return {
        "ela_comp_max_z": float(np.abs(z_ela).max()),
        "ela_comp_frac_out": float((ela > med + 3 * scale).mean()) if scale > 0 else 0.0,
        "noise_comp_max_z": float(np.abs(_robust_z(noise)[0]).max()),
        "edge_comp_max_z": float(np.abs(_robust_z(edge)[0]).max()),
        "intensity_comp_max_z": float(np.abs(_robust_z(intensity)[0]).max()),
    }


def load_or_extract_v2(samples: list, cache_path, processes: int = 10) -> pd.DataFrame:
    """Tabela com nome da imagem e as features v2. Lê de `cache_path` (CSV) se existir;
    senão calcula em paralelo e salva."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_csv(cache_path)
    with get_context("fork").Pool(processes, initializer=_init_worker) as pool:
        rows = pool.map(component_features, [s.image_path for s in samples])
    table = pd.DataFrame(rows, columns=V2_NAMES)
    table.insert(0, "image", [s.image_path.name for s in samples])
    table.to_csv(cache_path, index=False)
    return table
