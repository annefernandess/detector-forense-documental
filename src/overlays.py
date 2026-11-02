"""
Imagens sobrepostas da interface: ELA, JPEG Ghost e copy-move desenhados sobre a imagem
original, com opacidade ajustável. Usam as mesmas paletas de `visualize_ela` (inferno) e
`visualize_jpeg_ghost` (tab10, blocos lisos sem cor). Nada aqui recebe anotações, exceto
`desenhar_gabarito`, que existe só para o modo demonstração.
"""

from __future__ import annotations

import cv2
import numpy as np
from matplotlib import colormaps

MAX_PARES = 30
AZUL, LARANJA, AMARELO, VERMELHO = (31, 119, 255), (255, 140, 0), (255, 220, 0), (230, 30, 30)

LEGENDAS = {
    "original": "imagem sem sobreposição, como referência",
    "ela": "amarelo e vermelho = maior diferença de compressão; sem cor = diferença baixa (escala ajustada a cada imagem: compara partes da mesma página)",
    "ghost": ("cor = qualidade JPEG de menor erro no bloco (azul 50, laranja 55, verde 60, vermelho 65, "
              "roxo 70, marrom 75, rosa 80, cinza 85, oliva 90); blocos sem texto ficam sem cor"),
    "copy_move": "azul e laranja = os dois trechos de um possível par; amarelo = linha que liga o par",
}


def _rgb(cinza: np.ndarray) -> np.ndarray:
    return np.repeat(cinza[..., None], 3, axis=2).astype(np.float32)


def _mesclar(base: np.ndarray, cor: np.ndarray, opacidade: float, mascara: np.ndarray | None = None) -> np.ndarray:
    """base*(1 − a) + cor*a, onde a = opacidade (e zero fora da máscara). Retorna uint8."""
    peso = np.full(base.shape[:2], opacidade, np.float32)
    if mascara is not None:
        peso = peso * mascara
    return np.clip(base * (1 - peso[..., None]) + cor * peso[..., None], 0, 255).astype(np.uint8)


def painel_original(cinza: np.ndarray) -> np.ndarray:
    """A imagem em RGB, sem sobreposição."""
    return _rgb(cinza).astype(np.uint8)


def painel_ela(cinza: np.ndarray, mapa_ela: np.ndarray, opacidade: float) -> np.ndarray:
    """
    Mapa de erro da ELA (inferno) sobre a imagem. Só de exibição: o mapa é normalizado pelo
    percentil 99,5 da própria imagem, e a opacidade de cada pixel é proporcional ao erro, para
    que áreas de erro baixo mostrem a imagem sem cobertura. `opacidade` (0 a 1) escala o conjunto.
    """
    topo = max(float(np.percentile(mapa_ela, 99.5)), 1.0)
    norma = np.clip(mapa_ela / topo, 0, 1).astype(np.float32)
    cor = colormaps["inferno"](norma)[..., :3] * 255
    return _mesclar(_rgb(cinza), cor.astype(np.float32), opacidade, norma)


def painel_ghost(cinza: np.ndarray, ghost_map: np.ndarray, escala: float, opacidade: float,
                 qualidades: tuple, tamanho_bloco: int = 16) -> np.ndarray:
    """
    Qualidade preferida por bloco (tab10) sobre a imagem. `ghost_map` está na escala da imagem
    original (blocos de `tamanho_bloco` px); `escala` leva à largura de exibição. Blocos lisos
    (valor −1) e a borda que não forma bloco completo ficam sem cor.
    """
    alto, largo = cinza.shape
    indices = np.full((alto, largo), -1, np.int16)
    lado = tamanho_bloco * escala
    ampliado = cv2.resize(ghost_map.astype(np.int16), (round(ghost_map.shape[1] * lado), round(ghost_map.shape[0] * lado)),
                          interpolation=cv2.INTER_NEAREST)
    h, w = min(alto, ampliado.shape[0]), min(largo, ampliado.shape[1])
    indices[:h, :w] = ampliado[:h, :w]
    paleta = np.array(colormaps["tab10"].colors[:len(qualidades)], np.float32) * 255
    cor = paleta[np.clip(indices, 0, len(qualidades) - 1)]
    return _mesclar(_rgb(cinza), cor, opacidade, (indices >= 0).astype(np.float32))


def painel_copy_move(cinza: np.ndarray, pares: list, escala: float, opacidade: float) -> np.ndarray:
    """
    Linhas ligando cada par de possível repetição e as duas caixas em cores diferentes (azul e
    laranja). A busca não sabe qual lado é a origem. Mostra no máximo MAX_PARES pares, os de
    maior correlação.
    """
    base = _rgb(cinza)
    camada = np.zeros(base.shape, np.uint8)

    def caixa(b):
        x, y, w, h = (round(v * escala) for v in b)
        return x - 2, y - 2, max(w, 6) + 4, max(h, 6) + 4  # folga de 2 px, só para a marcação ficar visível

    desenho = [(caixa(a), caixa(b)) for a, b, _ in sorted(pares, key=lambda p: -p[2])[:MAX_PARES]]
    for (ax, ay, aw, ah), (bx, by, bw, bh) in desenho:
        cv2.line(camada, (ax + aw // 2, ay + ah // 2), (bx + bw // 2, by + bh // 2), AMARELO, 2)
    for (ax, ay, aw, ah), (bx, by, bw, bh) in desenho:
        cv2.rectangle(camada, (ax, ay), (ax + aw, ay + ah), AZUL, 2)
        cv2.rectangle(camada, (bx, by), (bx + bw, by + bh), LARANJA, 2)
    return _mesclar(base, camada.astype(np.float32), opacidade, camada.any(axis=2).astype(np.float32))


def desenhar_gabarito(rgb: np.ndarray, regioes: list, escala: float) -> np.ndarray:
    """Caixas vermelhas das regiões anotadas (gabarito do dataset). Só para o modo demonstração."""
    saida = rgb.copy()
    for r in regioes:
        if r.is_original_area is True:
            continue
        x, y, w, h = (round(v * escala) for v in r.bbox)
        cv2.rectangle(saida, (x, y), (x + max(w, 6), y + max(h, 6)), VERMELHO, 2)
    return saida


def recortar(rgb: np.ndarray, x: float, y: float, largura: float, altura: float) -> np.ndarray:
    """Recorte por frações da imagem (0 a 1): canto superior esquerdo (x, y), largura e altura."""
    alto, largo = rgb.shape[:2]
    x0, y0 = int(x * largo), int(y * alto)
    return rgb[y0:max(y0 + 1, int((y + altura) * alto)), x0:max(x0 + 1, int((x + largura) * largo))]
