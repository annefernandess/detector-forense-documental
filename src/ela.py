"""
Error Level Analysis (ELA): recomprime a imagem em JPEG numa qualidade conhecida
e mede a diferença em relação à original. Regiões editadas tendem a ter um "nível
de erro" diferente do resto, por terem histórico de compressão distinto.
Fundamentação teórica (DCT, quantização, limitações): report/fundamentacao_ela.md
Referência: Krawetz, N. "A Picture's Worth: Digital Image Analysis and Forensics" (2007).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageChops


def _load_rgb(image_path) -> Image.Image:
    """Abre a imagem como RGB. Canais alfa (RGBA/LA) são compostos sobre
    fundo branco, para pixels transparentes não virarem lixo de cor."""
    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert("RGB")


def compute_ela(
    image_path: str,
    quality: int = 90,
    amplification: float = 15,
    grayscale: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Calcula o mapa de ELA e um score escalar da imagem inteira.

    Parâmetros:
        image_path: caminho da imagem a analisar
        quality: qualidade de recompressão JPEG usada como referência (testar
                 valores entre 70-95 e ver qual separa melhor autêntico/adulterado)
        amplification: fator que multiplica a diferença bruta antes do clip
                       em [0, 255]. Só afeta o mapa, não o score.
        grayscale: se True, faz a análise só na luminância (modo 'L'). Se False
                   (padrão), recomprime em RGB e usa o maior erro entre os 3 canais.

    Retorna:
        ela_map: array uint8 (H, W) com a diferença amplificada e limitada a
                 [0, 255], pronto para ser exibido como mapa de calor.
        score:   percentil 95 da diferença BRUTA (não amplificada), em níveis
                 de cinza. Independe de `amplification` e serve como resumo
                 simples da imagem para a avaliação quantitativa.
    """
    original = _load_rgb(image_path)
    if grayscale:
        original = original.convert("L")

    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)
    recompressed.load()

    diff = np.asarray(ImageChops.difference(original, recompressed), dtype=np.float32)
    if diff.ndim == 3:
        # (H, W, 3) -> (H, W): maior erro entre os canais em cada pixel
        diff = diff.max(axis=2)

    score = float(np.percentile(diff, 95))
    ela_map = np.clip(diff * amplification, 0, 255).astype(np.uint8)
    return ela_map, score


def visualize_ela(
    image_path,
    ela_map: np.ndarray,
    regions: list | None = None,
    score: float | None = None,
    title: str | None = None,
    axes=None,
    cmap: str = "inferno",
):
    """
    Plota, em 1 linha x 3 colunas:
        [0] imagem original
        [1] mapa de ELA (colormap `cmap`)
        [2] mapa de ELA sobreposto à imagem, com as regiões anotadas

    `regions` é a lista de TamperRegion de um DocumentSample (pode ser vazia
    ou None para autênticos). As regiões modificadas aparecem como retângulos
    vermelhos. As marcadas como área original (fonte do copy-paste,
    is_original_area=True) aparecem em ciano tracejado.

    Se `axes` for None, cria uma figura nova. Retorna a figura.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    else:
        fig = axes[0].figure

    original = _load_rgb(image_path)

    axes[0].imshow(original)
    axes[0].set_title(title or "Original", fontsize=9)

    axes[1].imshow(ela_map, cmap=cmap, vmin=0, vmax=255)
    axes[1].set_title("ELA" + (f" (score p95 = {score:.2f})" if score is not None else ""),
                      fontsize=9)

    axes[2].imshow(original.convert("L"), cmap="gray")
    axes[2].imshow(ela_map, cmap=cmap, vmin=0, vmax=255, alpha=0.6)
    axes[2].set_title("ELA sobreposta + regiões anotadas", fontsize=9)

    for r in regions or []:
        x, y, w, h = r.bbox
        is_source = getattr(r, "is_original_area", None) is True
        axes[2].add_patch(Rectangle(
            (x, y), w, h, fill=False,
            edgecolor="cyan" if is_source else "red",
            linestyle="--" if is_source else "-",
            linewidth=1.5,
        ))

    for ax in axes:
        ax.axis("off")
    return fig


def compute_region_score(image_path: str, diff_map: np.ndarray, regions: list) -> tuple[float, float]:
    """
    Compara a média do erro ELA nas regiões anotadas contra uma região de controle.
    Usa Otsu na imagem original para identificar pixels de texto e garantir que a
    região de controle também contenha texto.
    
    Retorna: (média_anotada, média_controle)
    """
    import cv2
    import random
    
    if not regions:
        return 0.0, 0.0
        
    # Carrega imagem em tons de cinza para Otsu
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0, 0.0
        
    # Otsu: THRESH_BINARY_INV assume texto escuro em fundo claro
    _, text_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Heurística: se a máscara ativou mais de 50% dos pixels, o texto devia ser claro no fundo escuro
    if np.count_nonzero(text_mask) > text_mask.size / 2:
        text_mask = cv2.bitwise_not(text_mask)
        
    H, W = diff_map.shape[:2]
    
    annotated_errors = []
    control_errors = []
    
    for r in regions:
        if getattr(r, "is_original_area", None) is True:
            continue  # Ignora áreas originais (fontes de copy-paste)
            
        x, y, w, h = r.bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        
        if x1 >= x2 or y1 >= y2:
            continue
            
        annotated_errors.append(diff_map[y1:y2, x1:x2].mean())
        
        # Sorteia região de controle
        max_attempts = 100
        control_mean = 0.0
        
        for _ in range(max_attempts):
            cx = random.randint(0, max(0, W - w))
            cy = random.randint(0, max(0, H - h))
            
            # Verifica sobreposição com qualquer região anotada
            overlap = False
            for ar in regions:
                ax, ay, aw, ah = ar.bbox
                if not (cx + w <= ax or cx >= ax + aw or cy + h <= ay or cy >= ay + ah):
                    overlap = True
                    break
                    
            if overlap:
                continue
                
            c_text = text_mask[cy:cy+h, cx:cx+w]
            if np.count_nonzero(c_text) > 0:
                control_mean = diff_map[cy:cy+h, cx:cx+w].mean()
                break
                
        control_errors.append(control_mean)
        
    if not annotated_errors:
        return 0.0, 0.0
        
    return float(np.mean(annotated_errors)), float(np.mean(control_errors))
