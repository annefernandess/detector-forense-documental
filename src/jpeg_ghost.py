"""
JPEG Ghost: recomprime a imagem em várias qualidades JPEG e, para cada bloco,
descobre em qual qualidade a diferença é mínima. Regiões coladas de outra fonte
tendem a "preferir" uma qualidade diferente da do resto da imagem.
Fundamentação teórica (quantização, por que surge o "fantasma", limitações):
report/fundamentacao_jpeg_ghost.md
Referência: Farid, H. "Exposing Digital Forgeries from JPEG Ghosts" (IEEE TIFS, 2009).
"""

from __future__ import annotations

import io
from collections import Counter

import numpy as np
from PIL import Image

# q=95 fica de fora do padrão: recomprimir quase sem perda dá erro pequeno para
# QUALQUER histórico, então o 95 "venceria" em quase todo bloco sem dizer nada
# sobre a compressão original (ver seção de limitações no .md).
DEFAULT_QUALITIES = tuple(range(50, 95, 5))  # 50, 55, ..., 90


def _load_gray(image) -> np.ndarray:
    """Abre (caminho ou PIL.Image) em tons de cinza float32. Canais alfa são
    compostos sobre fundo branco, como em ela.py."""
    img = image if isinstance(image, Image.Image) else Image.open(image)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return np.asarray(img.convert("L"), dtype=np.float32)


def _jpeg_roundtrip(gray: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer), dtype=np.float32)


def ghost_curves(image, qualities=DEFAULT_QUALITIES, block_size: int = 16):
    """
    Curvas de erro por bloco. Retorna (curves, block_std):
        curves:    (len(qualities), H_b, W_b) erro quadrático médio de cada bloco
                   para cada qualidade de recompressão
        block_std: (H_b, W_b) desvio padrão dos pixels de cada bloco (textura)
    Pixels que sobram nas bordas (H, W não múltiplos de block_size) são descartados.
    """
    gray = _load_gray(image)
    hb, wb = gray.shape[0] // block_size, gray.shape[1] // block_size
    gray = gray[:hb * block_size, :wb * block_size]

    curves = np.empty((len(qualities), hb, wb), dtype=np.float32)
    for i, q in enumerate(qualities):
        sq_err = (gray - _jpeg_roundtrip(gray, q)) ** 2
        curves[i] = sq_err.reshape(hb, block_size, wb, block_size).mean(axis=(1, 3))
    block_std = gray.reshape(hb, block_size, wb, block_size).std(axis=(1, 3))
    return curves, block_std


def compute_jpeg_ghost(
    image_path,
    qualities=DEFAULT_QUALITIES,
    block_size: int = 16,
    min_std: float = 10.0,
) -> np.ndarray:
    """
    Mapa espacial de "qual qualidade cada bloco prefere".

    Retorna array int (H // block_size, W // block_size) com o ÍNDICE (em
    `qualities`) da qualidade que minimizou o erro naquele bloco. Blocos quase
    lisos (desvio padrão < `min_std`, ex: papel em branco) valem -1: neles o
    erro é ~0 em todas as qualidades e o argmin não tem significado.
    """
    curves, block_std = ghost_curves(image_path, qualities, block_size)
    ghost_map = curves.argmin(axis=0).astype(np.int32)
    ghost_map[block_std < min_std] = -1
    return ghost_map


def has_jpeg_history(image_path, qualities=DEFAULT_QUALITIES) -> tuple[bool, int | None]:
    """
    Verifica se a imagem carrega vestígio de uma compressão JPEG anterior.

    Calcula o erro da imagem inteira para cada qualidade. Sem histórico JPEG,
    o erro só cai conforme a qualidade sobe (curva monotônica). Com histórico,
    aparece um vale (mínimo local interno) perto da qualidade original q0.
    Retorna (tem_vale, qualidade_do_vale_ou_None).
    """
    gray = _load_gray(image_path)
    err = np.array([((gray - _jpeg_roundtrip(gray, q)) ** 2).mean() for q in qualities])
    valleys = [i for i in range(1, len(err) - 1) if err[i] < err[i - 1] and err[i] < err[i + 1]]
    if not valleys:
        return False, None
    best = min(valleys, key=lambda i: err[i])
    return True, int(qualities[best])


def _region_block_mask(shape, regions, block_size, skip_original=True) -> np.ndarray:
    """Máscara (H_b, W_b) dos blocos cujo CENTRO cai dentro de alguma região."""
    hb, wb = shape
    centers_y = np.arange(hb) * block_size + block_size / 2
    centers_x = np.arange(wb) * block_size + block_size / 2
    mask = np.zeros(shape, dtype=bool)
    for r in regions or []:
        if skip_original and getattr(r, "is_original_area", None) is True:
            continue
        x, y, w, h = r.bbox
        rows = (centers_y >= y) & (centers_y < y + h)
        cols = (centers_x >= x) & (centers_x < x + w)
        mask |= rows[:, None] & cols[None, :]
    return mask


def _mode(values: np.ndarray) -> int | None:
    values = values[values >= 0]
    if values.size == 0:
        return None
    return Counter(values.tolist()).most_common(1)[0][0]


def compute_region_ghost_score(
    ghost_map: np.ndarray,
    regions: list,
    block_size: int = 16,
) -> tuple[int | None, int | None]:
    """
    Compara a qualidade "dominante" (moda dos índices) dentro das regiões
    anotadas com a do resto da imagem.

    - Dentro: blocos válidos cujo centro cai numa região modificada
      (regiões com is_original_area=True, fonte do copy-paste, são ignoradas).
    - Controle: blocos válidos fora de TODAS as regiões anotadas.

    Retorna (moda_dentro, moda_controle) como índices em `qualities`, ou None
    se não houver bloco válido. Modas diferentes = inconsistência de histórico.
    """
    inside = _region_block_mask(ghost_map.shape, regions, block_size, skip_original=True)
    any_region = _region_block_mask(ghost_map.shape, regions, block_size, skip_original=False)
    return _mode(ghost_map[inside]), _mode(ghost_map[~any_region])


def visualize_jpeg_ghost(
    ghost_map: np.ndarray,
    image,
    regions: list | None = None,
    qualities=DEFAULT_QUALITIES,
    block_size: int = 16,
    title: str | None = None,
    axes=None,
):
    """
    Plota, em 1 linha x 2 colunas:
        [0] imagem original
        [1] mapa de JPEG Ghost (colormap categórico, uma cor por qualidade;
            blocos lisos (-1) ficam transparentes sobre a imagem em cinza)
    Regiões anotadas são desenhadas nos dois painéis, no mesmo estilo de
    visualize_ela: modificadas em vermelho, áreas originais em ciano tracejado.

    Se `axes` for None, cria uma figura nova. Retorna a figura.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Rectangle

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    else:
        fig = axes[0].figure

    gray = _load_gray(image)
    hb, wb = ghost_map.shape
    extent = (0, wb * block_size, hb * block_size, 0)  # mapa na escala de pixels

    n = len(qualities)
    cmap = ListedColormap(plt.get_cmap("tab10").colors[:n])
    norm = BoundaryNorm(np.arange(-0.5, n), n)

    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(title or "Original", fontsize=9)

    axes[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    im = axes[1].imshow(np.ma.masked_less(ghost_map, 0), cmap=cmap, norm=norm,
                        extent=extent, interpolation="nearest", alpha=0.75)
    axes[1].set_title("JPEG Ghost: qualidade de menor erro por bloco", fontsize=9)
    cbar = fig.colorbar(im, ax=axes[1], ticks=range(n), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([str(q) for q in qualities])

    for ax in axes:
        for r in regions or []:
            x, y, w, h = r.bbox
            is_source = getattr(r, "is_original_area", None) is True
            ax.add_patch(Rectangle(
                (x, y), w, h, fill=False,
                edgecolor="cyan" if is_source else "red",
                linestyle="--" if is_source else "-",
                linewidth=1.5,
            ))
        ax.set_xlim(0, gray.shape[1])
        ax.set_ylim(gray.shape[0], 0)
        ax.axis("off")
    return fig
