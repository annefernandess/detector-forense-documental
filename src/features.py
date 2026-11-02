"""
Features por imagem para o classificador de adulteração (notebook 04). Nenhuma
feature usa as anotações (regiões, caixas, is_original_area): todas funcionam em
uma imagem nova, sem gabarito. Tudo em tons de cinza, com o canal alfa composto
sobre fundo branco (`load_gray`), para que o modo do arquivo (L x RGBA) não vire sinal.
"""

from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from copy_move import blind_copy_move
from dataset_loader import load_gray
from ela import compute_ela
from jpeg_ghost import DEFAULT_QUALITIES, compute_jpeg_ghost, has_jpeg_history

FEATURE_NAMES = ["ela_p95", "copy_move_peak", "ghost_q90_frac", "has_jpeg_history",
                 "noise_std", "noise_p95", "edge_std"]

COPY_MOVE_FLOOR = 0.8  # pico registrado quando nenhum par de componentes passa de 0,8


def _text_mask(gray: np.ndarray) -> np.ndarray:
    """Máscara de Otsu do texto (escuro sobre claro), calculada sobre `load_gray`.
    Otsu não aparece nos slides; a limiarização aparece nos slides 132, 179, 185 e 231."""
    _, mask = cv2.threshold(gray.astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(mask) > mask.size / 2:
        mask = cv2.bitwise_not(mask)
    return mask > 0


def ela_p95(image_path) -> float:
    """Percentil 95 do erro da ELA na imagem inteira (qualidade 90, tons de cinza).
    Base: DCT (slides 376-379 e 432-440), quantização (127 e 133) e JPEG (157)."""
    return compute_ela(image_path, quality=90, grayscale=True)[1]


def copy_move_peak(image_path) -> float:
    """Maior correlação entre componentes de texto de mesmo tamanho e distantes entre si,
    com a escala e a lógica da busca cega (`blind_copy_move`). Valores abaixo de 0,8 são
    registrados como 0,8. Base: correlação normalizada e filtro casado (slides 260-272)."""
    pairs, _ = blind_copy_move(load_gray(image_path), COPY_MOVE_FLOOR)
    return max((r for _, _, r in pairs), default=COPY_MOVE_FLOOR)


def ghost_features(image_path) -> tuple[float, int]:
    """(fração de blocos de texto com argmin de qualidade em 90, has_jpeg_history).
    Sem histórico JPEG, retorna (0, 0). Base: quantização (slides 127 e 133), DCT
    (376-379 e 432-440) e JPEG (157)."""
    has_history, _ = has_jpeg_history(image_path)
    if not has_history:
        return 0.0, 0
    ghost_map = compute_jpeg_ghost(image_path)
    valid = ghost_map[ghost_map >= 0]  # blocos lisos (sem texto) valem -1
    top = len(DEFAULT_QUALITIES) - 1
    return (float((valid == top).mean()) if valid.size else 0.0), 1


def noise_residual(image_path) -> tuple[float, float]:
    """(desvio padrão do resíduo, percentil 95 do resíduo absoluto) sobre pixels de texto,
    com resíduo = imagem − filtro da mediana 3x3. Base: filtro da mediana (slides 201 e
    208-216)."""
    gray = load_gray(image_path)
    u8 = np.clip(gray, 0, 255).astype(np.uint8)
    residual = u8.astype(np.float32) - cv2.medianBlur(u8, 3).astype(np.float32)
    values = residual[_text_mask(gray)]
    if values.size == 0:
        return 0.0, 0.0
    return float(values.std()), float(np.percentile(np.abs(values), 95))


def edge_std(image_path) -> float:
    """Desvio padrão da magnitude do gradiente de Sobel nos pixels de texto (máscara de
    Otsu), como medida de consistência de traço. Base: gradiente e Sobel (slides 225-230)."""
    gray = load_gray(image_path)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    values = np.hypot(gx, gy)[_text_mask(gray)]
    return float(values.std()) if values.size else 0.0


def extract_features(image_path) -> dict:
    """Todas as features de uma imagem, na ordem de FEATURE_NAMES."""
    ghost_frac, has_history = ghost_features(image_path)
    noise_std, noise_p95 = noise_residual(image_path)
    return {"ela_p95": ela_p95(image_path), "copy_move_peak": copy_move_peak(image_path),
            "ghost_q90_frac": ghost_frac, "has_jpeg_history": has_history,
            "noise_std": noise_std, "noise_p95": noise_p95, "edge_std": edge_std(image_path)}


def _init_worker():
    cv2.setNumThreads(1)  # 1 thread por processo: evita disputa de CPU entre os processos


def load_or_extract(samples: list, cache_path, processes: int = 10) -> pd.DataFrame:
    """
    Tabela com uma linha por imagem: nome, rótulo (`is_tampered`, da coluna `forged`),
    tipos de adulteração (só para análise, nunca como feature) e as features. Lê de
    `cache_path` (CSV) se existir; senão calcula em paralelo e salva.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_csv(cache_path)
    with get_context("fork").Pool(processes, initializer=_init_worker) as pool:
        rows = pool.map(extract_features, [s.image_path for s in samples])
    table = pd.DataFrame(rows, columns=FEATURE_NAMES)
    table.insert(0, "image", [s.image_path.name for s in samples])
    table.insert(1, "is_tampered", [int(s.is_tampered) for s in samples])
    table.insert(2, "tamper_types", ["+".join(s.tamper_types) for s in samples])
    table.to_csv(cache_path, index=False)
    return table
