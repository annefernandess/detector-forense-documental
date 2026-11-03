"""
Monta o conjunto sintético de copy-move a partir das notas do CORD (data/cord/imagens/).
Notas em que o gerador congelado não acha par origem/destino ficam fora das duas classes
(excluí-las só da classe adulterada enviesaria a comparação). As demais são divididas ao
meio, com seed 42, e cada nota entra em uma só classe. Grava PNG e inventario.csv.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from gerar_copy_move_sintetico import SEED, colar_caixa_texto, fracao_texto, mascara_texto


def montar_conjunto(origem: Path, destino: Path) -> dict:
    """
    Lê os PNG de `origem` (em tons de cinza), tenta gerar a adulteração em todos, divide as
    geráveis em duas metades (adulterada/autêntica) e grava em `destino` com inventario.csv
    (arquivo, classe, x, y, largura, altura, nota_origem). Devolve contagens e verificações.
    """
    destino.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    geradas, excluidas = {}, []
    for caminho in sorted(origem.glob("*.png")):
        gray = np.asarray(Image.open(caminho).convert("L"))
        try:
            geradas[caminho.name] = (gray, *colar_caixa_texto(gray, rng))
        except ValueError:
            excluidas.append(caminho.name)

    nomes = sorted(geradas)
    escolhidas = set(np.random.default_rng(SEED).permutation(nomes)[:len(nomes) // 2])
    linhas, hashes, fracoes = [], [], []
    for nome in nomes:
        gray, editada, caixa = geradas[nome]
        adulterada = nome in escolhidas
        saida = editada if adulterada else gray
        arquivo = ("adulterada_" if adulterada else "autentica_") + nome
        Image.fromarray(saida).save(destino / arquivo)
        hashes.append(hashlib.md5(saida.tobytes()).hexdigest())
        if adulterada:
            fracoes.append(fracao_texto(mascara_texto(gray), caixa))  # texto na caixa antes da colagem
        linhas.append([arquivo, "adulterada" if adulterada else "autentica",
                       *(caixa if adulterada else ("", "", "", "")), nome])
    with open(destino / "inventario.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "classe", "x", "y", "largura", "altura", "nota_origem"])
        w.writerows(linhas)
    return {"n_adulteradas": len(escolhidas), "n_autenticas": len(nomes) - len(escolhidas),
            "excluidas_sem_par": excluidas, "imagens_duplicadas": len(hashes) - len(set(hashes)),
            "fracao_texto_caixa_min": float(min(fracoes)), "fracao_texto_caixa_media": float(np.mean(fracoes))}


def extrair_parquets(parquets: list, destino: Path, prefixo: str) -> int:
    """Grava os bytes das imagens dos parquet (em ordem) como `<prefixo>_NNNN.png`, sem alterar o conteúdo."""
    destino.mkdir(parents=True, exist_ok=True)
    n = 0
    for caminho in parquets:
        for lote in pq.ParquetFile(caminho).iter_batches(batch_size=50, columns=["image"]):
            for imagem in lote.column("image").to_pylist():
                (destino / f"{prefixo}_{n:04d}.png").write_bytes(imagem["bytes"])
                n += 1
    return n


def montar_desenvolvimento(origem: Path, destino: Path) -> dict:
    """
    Conjunto de desenvolvimento: aplica o gerador (seed 42) a cada nota de `origem` e grava só as
    adulteradas, com inventario.csv (arquivo, x, y, largura, altura, nota_origem). Localização não
    precisa de notas autênticas. Notas sem par origem/destino ficam de fora e são listadas.
    """
    destino.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    linhas, excluidas = [], []
    for caminho in sorted(origem.glob("*.png")):
        gray = np.asarray(Image.open(caminho).convert("L"))
        try:
            editada, caixa = colar_caixa_texto(gray, rng)
        except ValueError:
            excluidas.append(caminho.name)
            continue
        arquivo = "adulterada_" + caminho.name
        Image.fromarray(editada).save(destino / arquivo)
        linhas.append([arquivo, "adulterada", *caixa, caminho.name])
    with open(destino / "inventario.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "classe", "x", "y", "largura", "altura", "nota_origem"])
        w.writerows(linhas)
    return {"n_adulteradas": len(linhas), "excluidas_sem_par": excluidas}
