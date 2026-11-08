"""
Diagnóstico temporário (só train): uma janela de deslocamento maior que +-2 px eleva a correlação do
resíduo (k=3) das cópias sem elevar a dos pares de autênticas? Usa funções de src/ sem alterá-las.
Rodar na raiz: python notebooks/diagnostico_desalinhamento_train.py
"""
import hashlib
import json
import sys
import warnings
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import cv2
import numpy as np

from copy_move import blind_copy_move, candidate_boxes, correlation_peak, filter_pairs, region_covered
from copy_move_residuo import MIN_SIDE, correlacao_residuo, residuo
from dataset_loader import load_dataset, load_gray
from evaluation import wilson_score_interval  # noqa: F401  (não usado; mantém o padrão de imports do projeto)
from copy_move_residuo import separacao_auc

DS = (2, 4, 6, 8)
K = 3
LIMIAR_PICO = 0.979


def trabalho(argumento):
    """Uma imagem: devolve linhas dos grupos (a) regiões CPI vs pico, (b)/(c) pares da busca atual."""
    cv2.setNumThreads(1)
    tipo, caminho, caixas = argumento
    gray = load_gray(caminho)
    res = residuo(gray, K)
    linhas = []
    if tipo == "cpi":
        for caixa in caixas:
            pico, loc = correlation_peak(gray, caixa)
            if pico is None:
                continue
            origem = (max(0, caixa[0]), max(0, caixa[1]), loc[2], loc[3])
            linhas.append({"grupo": "a", "pico": pico, "largura_imagem": gray.shape[1],
                           **{f"d{d}": correlacao_residuo(res, origem, loc, d) for d in DS}})
    pares, _ = blind_copy_move(gray, 0.95, min_side=MIN_SIDE)
    mantidos = filter_pairs(pares, 0.99)
    boxes_por_par = [(p, candidate_boxes([p])) for p in mantidos]
    for par, boxes in boxes_por_par:
        base = {"largura_imagem": gray.shape[1], "intensidade": par[2],
                **{f"d{d}": correlacao_residuo(res, par[0], par[1], d) for d in DS}}
        if tipo == "aut":
            linhas.append({"grupo": "b", **base})
        elif any(region_covered(c, boxes, 0.5) for c in caixas):
            linhas.append({"grupo": "c", **base})  # par da busca atual que cobre uma região CPI
    return linhas


def estatisticas(valores):
    v = np.array([x for x in valores if x is not None])
    return {"n": int(len(v)), "n_indefinidos": int(len(valores) - len(v)), "mediana": float(np.median(v)),
            "fracao_ge_0.9": float((v >= 0.9).mean())}


if __name__ == "__main__":
    train = [s for s in load_dataset("data/findit2", "train") if s.image_path.exists()]
    tarefas = []
    for s in train:
        if s.is_tampered and s.regions and "CPI" in s.tamper_types:
            caixas = [r.bbox for r in s.regions if r.is_original_area is not True and r.modification and "CPI" in r.modification]
            if caixas:
                tarefas.append(("cpi", str(s.image_path), caixas))
        elif not s.is_tampered:
            tarefas.append(("aut", str(s.image_path), []))
    with get_context("fork").Pool(8) as pool:
        linhas = [l for ls in pool.map(trabalho, tarefas) for l in ls]

    grupos = {"a": [l for l in linhas if l["grupo"] == "a"], "b": [l for l in linhas if l["grupo"] == "b"],
              "c": [l for l in linhas if l["grupo"] == "c"]}
    grupos["a'"] = [l for l in grupos["a"] if l["pico"] >= LIMIAR_PICO]
    tabela = {g: {f"d{d}": estatisticas([l[f"d{d}"] for l in ls]) for d in DS} for g, ls in grupos.items()}
    auc = {}
    for g in ("a", "a'"):
        auc[g] = {}
        for d in DS:
            a = [l[f"d{d}"] for l in grupos[g] if l[f"d{d}"] is not None]
            b = [l[f"d{d}"] for l in grupos["b"] if l[f"d{d}"] is not None]
            auc[g][f"d{d}"] = separacao_auc(a, b)

    decisao = {}
    for g in ("a", "a'"):
        medianas = {d: tabela[g][f"d{d}"]["mediana"] for d in DS}
        melhor = max(DS, key=lambda d: medianas[d])
        ganho = medianas[melhor] - medianas[2]
        subida_b = tabela["b"][f"d{melhor}"]["fracao_ge_0.9"] - tabela["b"]["d2"]["fracao_ge_0.9"]
        decisao[g] = {"melhor_d_pela_mediana": melhor, "ganho_mediana_a": ganho, "subida_fracao_b_pontos": 100 * subida_b,
                      "criterio_atendido": bool(ganho >= 0.15 and subida_b < 0.05)}

    codigo = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    saida = {"escopo": "só train do Find it again!; val e test não abertos", "k": K, "deslocamentos": list(DS),
             "grupos": {"a": "regiões CPI anotadas vs pico de correlação", "a'": f"(a) com pico >= {LIMIAR_PICO}",
                        "b": "pares da busca atual (piso 0,99) em autênticas",
                        "c": "(extra, fora da regra de decisão) pares da busca atual (piso 0,99) em imagens CPI cujas caixas cobrem >= 50% de uma região CPI"},
             "tabela": tabela, "auc_a_contra_b": auc, "decisao_pre_combinada": decisao,
             "largura_mediana_imagens_c": float(np.median([l["largura_imagem"] for l in grupos["c"]])) if grupos["c"] else None,
             "codigo": "notebooks/diagnostico_desalinhamento_train.py", "codigo_sha256": codigo}
    Path("results/diagnostico_desalinhamento_train.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    for g in ("a", "a'", "b", "c"):
        for d in DS:
            t = tabela[g][f"d{d}"]
            print(f"{g:3} d={d} n={t['n']:4} indef={t['n_indefinidos']:2} mediana={t['mediana']:.3f} >=0,9: {t['fracao_ge_0.9']:.1%}"
                  + (f" AUC={auc[g][f'd{d}']:.3f}" if g in auc else ""))
    print(json.dumps(decisao, ensure_ascii=False, indent=1))
    print("largura mediana (c):", saida["largura_mediana_imagens_c"], "| sha256:", codigo)
