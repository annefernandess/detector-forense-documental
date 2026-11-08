"""
Ruído local por bloco, revisado: desvio-padrão (e não MAD) do resíduo, com piso, e z calibrado pelas
autênticas do train. Nenhuma anotação entra nos mapas nem nas features.

Resíduo = imagem - mediana 3 x 3 (copy_move_residuo.residuo), em tons de cinza, na resolução original.
Fundo = fora da máscara de Otsu do texto dilatada de 3 px (elemento 7 x 7; dilatação é método de apoio e
não aparece nos slides). Blocos inteiros de lado B; ignora-se o bloco com menos de LIMIAR_FUNDO de fundo.
Valor do bloco = log(sigma + PISO_SIGMA), com sigma = desvio-padrão (ddof = 0) do resíduo nos pixels de
fundo do bloco; o piso evita log(0) em papel plano. Calibração: média e desvio do valor nos blocos válidos
das autênticas do train, por B. Mapa z = (valor - media) / desvio. Feature de imagem: fração de blocos
válidos com |z| > LIMIAR_Z.

Sem multiprocessing (o cálculo é rápido). Grade e critérios em results/grade_ruido_local.json.
"""

from __future__ import annotations

import cv2
import numpy as np

from copy_move_residuo import residuo
from gerar_copy_move_sintetico import mascara_texto
from ruido_fundo import LIMIAR_FUNDO, LIMIAR_Z, TAMANHOS, _blocos

DILATACAO_PX = 3
PISO_SIGMA = 0.1


def mascara_fundo(gray: np.ndarray) -> np.ndarray:
    """Pixels de fundo: fora da máscara de Otsu do texto dilatada de DILATACAO_PX (elemento 7 x 7)."""
    lado = 2 * DILATACAO_PX + 1
    return cv2.dilate(mascara_texto(gray).astype(np.uint8), np.ones((lado, lado), np.uint8)) == 0


def valores_ruido_local(gray: np.ndarray, b: int) -> np.ndarray | None:
    """Mapa (ny, nx) de log(sigma + PISO_SIGMA) por bloco, NaN nos blocos com pouco fundo; None sem bloco válido."""
    fundo = mascara_fundo(gray)
    rb, fb = _blocos(residuo(gray, 3), b), _blocos(fundo, b)
    valido = fb.mean(axis=2) >= LIMIAR_FUNDO
    if not valido.any():
        return None
    r, f = rb[valido], fb[valido].astype(np.float32)
    n = f.sum(axis=1)
    media = (r * f).sum(axis=1) / n
    sigma = np.sqrt((((r - media[:, None]) ** 2) * f).sum(axis=1) / n)
    mapa = np.full(valido.shape, np.nan, np.float32)
    mapa[valido] = np.log(sigma + PISO_SIGMA)
    return mapa


def calibrar(mapas_de_valores: list) -> tuple:
    """(média, desvio) dos valores nos blocos válidos de todos os mapas dados (maps None são ignorados)."""
    v = np.concatenate([m[~np.isnan(m)] for m in mapas_de_valores if m is not None])
    return float(v.mean()), float(v.std())


def mapa_z(valores: np.ndarray | None, media: float, desvio: float) -> np.ndarray | None:
    """z = (valor - média) / desvio em cada bloco válido; None se o mapa de valores for None."""
    return None if valores is None else (valores - media) / desvio


def fracao_suspeita(z: np.ndarray | None) -> float | None:
    """Fração dos blocos válidos com |z| > LIMIAR_Z; None sem bloco válido."""
    if z is None:
        return None
    validos = np.abs(z[~np.isnan(z)])
    return float((validos > LIMIAR_Z).mean()) if len(validos) else None
