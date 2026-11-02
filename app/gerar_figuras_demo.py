"""
Gera figuras de 4 painéis (original, ELA, JPEG Ghost, copy-move) para uma imagem adulterada e
uma autêntica do val, com o mesmo pipeline e as mesmas funções de desenho da interface. As
imagens são escolhidas com seed 42, sem selecionar casos favoráveis. Saída: results/figuras_demo/
(ignorada pelo git, porque contém imagens do dataset). Rodar na raiz: python app/gerar_figuras_demo.py
"""

from __future__ import annotations

import random
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_loader import load_dataset
from overlays import LEGENDAS, painel_copy_move, painel_ela, painel_ghost, painel_original
from pipeline import analisar_imagem

OPACIDADE = 0.5
SAIDA = ROOT / "results" / "figuras_demo"


def figura(amostra, rotulo: str) -> Path:
    r = analisar_imagem(amostra.image_path)
    cinza, escala = r["cinza_exibicao"], r["escala"]
    if r["ghost_historico"]:
        ghost = painel_ghost(cinza, r["ghost_mapa"], escala, OPACIDADE, r["ghost_qualidades"])
        legenda_ghost = LEGENDAS["ghost"]
    else:
        ghost, legenda_ghost = painel_original(cinza), "não se aplica: sem histórico JPEG"
    pares = r["copy_move_pares"]
    paineis = [
        ("Original", painel_original(cinza), LEGENDAS["original"]),
        ("ELA: diferença de compressão", painel_ela(cinza, r["mapa_ela"], OPACIDADE), LEGENDAS["ela"]),
        ("JPEG Ghost: qualidade por bloco", ghost, legenda_ghost),
        ("Copy-move: possível repetição", painel_copy_move(cinza, pares, escala, OPACIDADE),
         LEGENDAS["copy_move"] if pares else "nenhum par encontrado"),
    ]
    fig, eixos = plt.subplots(1, 4, figsize=(22, 11))
    for eixo, (titulo, imagem, legenda) in zip(eixos, paineis):
        eixo.imshow(imagem)
        eixo.set_title(titulo, fontsize=12)
        eixo.set_xlabel("\n".join(textwrap.wrap(legenda, 44)), fontsize=9)
        eixo.set_xticks([])
        eixo.set_yticks([])
    fig.suptitle(f"{amostra.image_path.name} | gabarito do dataset (fora da análise): {rotulo} | "
                 f"faixa: {r['faixa']} | pares de copy-move: {len(pares)}", fontsize=13, y=0.985)
    fig.subplots_adjust(top=0.90, bottom=0.20, left=0.01, right=0.99, wspace=0.06)
    SAIDA.mkdir(parents=True, exist_ok=True)
    caminho = SAIDA / f"demo_{rotulo}_{amostra.image_path.stem}.png"
    fig.savefig(caminho, dpi=80)
    plt.close(fig)
    return caminho


if __name__ == "__main__":
    val = load_dataset(str(ROOT / "data/findit2"), "val")
    rng = random.Random(42)
    adulterada = rng.choice([s for s in val if s.is_tampered and "CPI" in s.tamper_types])
    autentica = rng.choice([s for s in val if not s.is_tampered])
    for amostra, rotulo in ((adulterada, "adulterada"), (autentica, "autentica")):
        print(figura(amostra, rotulo))
