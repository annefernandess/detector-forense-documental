"""
Calibra as caixas automáticas de região (src/regioes.py) usando SÓ o conjunto de validação; o test nunca é
lido. As anotações são usadas apenas aqui, para medir a precisão das caixas; nunca para gerar caixas nem na
interface. Grava results/regioes_config.json e results/regioes_calibracao.csv. O limite de "muitos pares" do
texto (bloco "explicacao" do JSON) vem de app/calibrar_limite_pares.py; este script o preserva.
Rodar na raiz do projeto: python app/calibrar_regioes.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import pandas as pd

from dataset_loader import load_dataset
from ela import compute_ela
from regioes import mapa_para_1000, regioes_ela

PERCENTIS = (95, 98, 99)
MAX_CAIXAS = (1, 2, 3)
CONTRASTES = (0.0, 2.0, 2.5, 3.0)  # níveis de cinza acima da mediana; 0 = sem filtro de contraste
LIMITE_FALSO_ALARME = 0.30
SORTEIOS = 50
SEED = 42


def _mapa(caminho):
    cv2.setNumThreads(1)
    bruto = compute_ela(caminho, quality=90, amplification=1, grayscale=True)[0]
    return os.path.basename(caminho), mapa_para_1000(bruto), bruto.shape[::-1]


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0


def _acerta(caixa, regioes) -> bool:
    """Acerto: IoU >= 0,1 com uma região modificada, ou centro da caixa dentro dela."""
    cx, cy = caixa[0] + caixa[2] / 2, caixa[1] + caixa[3] / 2
    return any(_iou(caixa, r.bbox) >= 0.1 or (r.bbox[0] <= cx <= r.bbox[0] + r.bbox[2]
                                              and r.bbox[1] <= cy <= r.bbox[1] + r.bbox[3]) for r in regioes)


def _em_original(caixas, largura):
    f = largura / 1000
    return [tuple(round(v * f) for v in c) for c, _ in caixas]


def medir(mapas, adulteradas, autenticas, p, m, k) -> dict:
    """Precisão por caixa, acerto por imagem (adulteradas), falso alarme (autênticas) e o acerto
    de um sorteio de `m` componentes entre os mesmos candidatos (referência do acaso)."""
    rng = np.random.default_rng(SEED)
    caixas_total = acertos = imagens_certas = 0
    acaso = 0.0
    for s in adulteradas:
        mapa, (largura, _) = mapas[s.image_path.name]
        modificadas = [r for r in s.regions if r.is_original_area is not True]  # regra de evaluation.py
        candidatas = _em_original(regioes_ela(mapa, p, 10 ** 6, k), largura)
        caixas = candidatas[:m]
        hits = [_acerta(c, modificadas) for c in caixas]
        caixas_total += len(caixas)
        acertos += sum(hits)
        imagens_certas += any(hits)
        if len(candidatas) <= m:
            acaso += float(any(_acerta(c, modificadas) for c in candidatas))
        else:
            acaso += np.mean([any(_acerta(candidatas[i], modificadas) for i in rng.choice(len(candidatas), m, replace=False))
                              for _ in range(SORTEIOS)])
    falso = sum(len(regioes_ela(mapas[s.image_path.name][0], p, m, k)) > 0 for s in autenticas)
    return {"percentil": p, "max_caixas": m, "contraste_min": k,
            "precisao": acertos / max(caixas_total, 1), "taxa_acerto": imagens_certas / len(adulteradas),
            "falso_alarme": falso / len(autenticas), "acerto_aleatorio": acaso / len(adulteradas),
            "caixas_por_imagem": caixas_total / len(adulteradas)}


def escolher(tabela: pd.DataFrame):
    """Maior taxa de acerto com falso alarme <= limite; empate: maior precisão, menos caixas, maior
    percentil. Se nenhuma combinação cumprir o limite, a de menor falso alarme (mesmo desempate)."""
    elegiveis = tabela[tabela["falso_alarme"] <= LIMITE_FALSO_ALARME]
    atingido = len(elegiveis) > 0
    if atingido:
        ordem = elegiveis.sort_values(["taxa_acerto", "precisao", "max_caixas", "percentil"],
                                      ascending=[False, False, True, False])
    else:
        ordem = tabela.sort_values(["falso_alarme", "taxa_acerto", "precisao", "max_caixas", "percentil"],
                                   ascending=[True, False, False, True, False])
    return ordem.iloc[0], atingido


if __name__ == "__main__":
    val = [s for s in load_dataset(str(ROOT / "data/findit2"), "val") if s.image_path.exists()]
    adulteradas = [s for s in val if s.is_tampered and s.regions]
    autenticas = [s for s in val if not s.is_tampered]
    caminhos = [s.image_path for s in val]
    with get_context("fork").Pool(10) as pool:
        mapas = {n: (m, wh) for n, m, wh in pool.map(_mapa, caminhos)}

    linhas = [medir(mapas, adulteradas, autenticas, p, m, k)
              for k, p, m in itertools.product(CONTRASTES, PERCENTIS, MAX_CAIXAS)]
    tabela = pd.DataFrame(linhas)
    melhor, atingido = escolher(tabela)

    destino = ROOT / "results/regioes_config.json"
    explicacao = json.loads(destino.read_text(encoding="utf-8"))["explicacao"]  # de calibrar_limite_pares.py; falha se faltar
    config = {
        "percentil": int(melhor["percentil"]), "max_caixas": int(melhor["max_caixas"]),
        "contraste_min": float(melhor["contraste_min"]),
        "precisao_val": float(melhor["precisao"]), "taxa_acerto_val": float(melhor["taxa_acerto"]),
        "falso_alarme_val": float(melhor["falso_alarme"]), "acerto_aleatorio_val": float(melhor["acerto_aleatorio"]),
        "limite_falso_alarme": LIMITE_FALSO_ALARME, "criterio_atingido": bool(atingido),
        "regra_acerto": "IoU >= 0,1 com região anotada modificada, ou centro da caixa dentro dela",
        "n_val": {"adulteradas": len(adulteradas), "autenticas": len(autenticas)},
        "explicacao": explicacao,
        "nota": "calibrado no val",
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    tabela.to_csv(ROOT / "results/regioes_calibracao.csv", index=False)
    destino.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(f"val: {len(adulteradas)} adulteradas, {len(autenticas)} autênticas | limite de falso alarme: {LIMITE_FALSO_ALARME:.0%}")
    print(tabela.round(3).to_string(index=False))
    print(f"\ncritério atingido: {atingido}")
    print(json.dumps(config, ensure_ascii=False, indent=2))
