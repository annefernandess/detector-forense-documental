from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def plot_sample_grid(authentic: list, forged: list):
    """
    Grade 2 x N: autênticas na linha de cima; adulteradas na de baixo, com as regiões
    modificadas em vermelho (as fontes de copy-paste não são desenhadas).
    Retorna a figura.
    """
    fig, axes = plt.subplots(2, len(authentic), figsize=(5 * len(authentic), 10))

    for ax, s in zip(axes[0], authentic):
        ax.imshow(Image.open(s.image_path).convert("RGB"))
        ax.set_title(f"Autêntico\n{s.image_path.name}")
        ax.axis("off")

    for ax, s in zip(axes[1], forged):
        ax.imshow(Image.open(s.image_path).convert("RGB"))
        ax.set_title(f"Adulterado\n{s.image_path.name}")
        ax.axis("off")
        for r in s.regions:
            if r.is_original_area is True:
                continue
            x, y, w, h = r.bbox
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))

    plt.tight_layout()
    return fig
