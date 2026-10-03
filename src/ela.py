"""
Error Level Analysis (ELA)

Ideia: recomprimir a imagem em JPEG numa qualidade conhecida e comparar
com a original. Regiões editadas tendem a ter um "nível de erro" diferente
do resto da imagem, porque foram salvas com histórico de compressão distinto.

Referência de partida: Krawetz, N. "A Picture's Worth: Digital Image Analysis
and Forensics" (2007).
"""

import numpy as np
from PIL import Image, ImageChops
import io


def compute_ela(image_path: str, quality: int = 90) -> np.ndarray:
    """
    Retorna um mapa de erro (mesma resolução da imagem original) que evidencia
    áreas com inconsistência de compressão.

    Parâmetros:
        image_path: caminho da imagem a analisar
        quality: qualidade de recompressão JPEG usada como referência (testar
                 valores entre 70-95 e ver qual separa melhor autêntico/adulterado)

    TODO:
        - carregar a imagem original
        - recomprimir em memória (io.BytesIO) na qualidade escolhida
        - calcular a diferença absoluta pixel a pixel (ImageChops.difference)
        - normalizar/amplificar a diferença para virar um mapa de calor visível
        - retornar o mapa como array numpy (H, W) ou (H, W, 3)
    """
    original = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    diff = ImageChops.difference(original, recompressed)

    # TODO: amplificar a diferença (multiplicar por um fator e usar np.clip)
    # para virar um mapa de calor visualmente interpretável
    diff_array = np.array(diff)

    return diff_array


def visualize_ela(diff_array: np.ndarray, amplification: int = 20):
    """
    TODO: usar matplotlib para plotar o mapa de calor da diferença,
    com uma colormap tipo 'hot' ou 'inferno'.
    """
    raise NotImplementedError
