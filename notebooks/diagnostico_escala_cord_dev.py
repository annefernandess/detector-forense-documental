"""
Diagnóstico temporário (só CORD de desenvolvimento): alargar a janela de alinhamento do resíduo (k=3)
recupera cópias exatas em digitalizações grandes? Usa funções de src/ sem alterá-las.
Rodar na raiz: python notebooks/diagnostico_escala_cord_dev.py
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

from avaliacao_sintetica import carregar_amostras
from copy_move import candidate_boxes, region_covered
from copy_move_residuo import candidatos, correlacao_residuo, filtrar_pares, residuo
from dataset_loader import load_gray

DS = (2, 4, 6, 8)
K = 3
PISO = 0.95
ESCALA_CORTE = 1.5


def trabalho(argumento):
    """Uma colagem: largura, nº de pares que cobrem a caixa colada e a maior correlação do resíduo por d."""
    cv2.setNumThreads(1)
    caminho, caixa = argumento
    gray = load_gray(caminho)
    pares = filtrar_pares(candidatos(gray, ks=(K,)), PISO)
    cobrem = [p for p in pares if region_covered(caixa, candidate_boxes([p[:3]]), 0.5)]
    res = residuo(gray, K)
    linha = {"largura": gray.shape[1], "escala": gray.shape[1] / 1000, "n_pares_cobrem": len(cobrem)}
    for d in DS:
        valores = [v for v in (correlacao_residuo(res, p[0], p[1], d) for p in cobrem) if v is not None]
        linha[f"d{d}"] = max(valores) if valores else None
    return linha


def resumo(linhas):
    """Cobertura e, por d, mediana e frações >= 0,9 (sobre as cobertas e sobre todas as colagens)."""
    cobertas = [l for l in linhas if l["d2"] is not None]
    saida = {"n_colagens": len(linhas), "n_cobertas": len(cobertas), "cobertura": len(cobertas) / len(linhas)}
    for d in DS:
        v = np.array([l[f"d{d}"] for l in cobertas])
        saida[f"d{d}"] = {"mediana": float(np.median(v)) if len(v) else None,
                          "n_ge_0.9": int((v >= 0.9).sum()),
                          "fracao_ge_0.9_sobre_cobertas": float((v >= 0.9).mean()) if len(v) else None,
                          "fracao_ge_0.9_sobre_todas": float((v >= 0.9).sum() / len(linhas))}
    return saida


if __name__ == "__main__":
    amostras = carregar_amostras(Path("data/cord_sintetico_dev"))
    with get_context("fork").Pool(8) as pool:
        linhas = pool.map(trabalho, [(str(s.image_path), s.regions[0].bbox) for s in amostras])
    estratos = {"todas": linhas, f"escala < {ESCALA_CORTE}": [l for l in linhas if l["escala"] < ESCALA_CORTE],
                f"escala >= {ESCALA_CORTE}": [l for l in linhas if l["escala"] >= ESCALA_CORTE]}
    tabela = {nome: resumo(ls) for nome, ls in estratos.items() if ls}

    melhor_d = max(DS, key=lambda d: tabela["todas"][f"d{d}"]["fracao_ge_0.9_sobre_cobertas"])
    sobe = lambda nome, base: 100 * (tabela[nome][f"d{melhor_d}"][base] - tabela[nome]["d2"][base])
    recuperadas = {nome: tabela[nome][f"d{melhor_d}"]["n_ge_0.9"] - tabela[nome]["d2"]["n_ge_0.9"]
                   for nome in tabela if nome != "todas"}
    total_rec = sum(recuperadas.values())
    pequena, grande = f"escala < {ESCALA_CORTE}", f"escala >= {ESCALA_CORTE}"
    concentrada = bool(total_rec > 0 and recuperadas.get(grande, 0) > total_rec / 2
                       and sobe(grande, "fracao_ge_0.9_sobre_cobertas") > sobe(pequena, "fracao_ge_0.9_sobre_cobertas"))
    decisao = {"melhor_d": melhor_d,
               "subida_pontos_sobre_cobertas": sobe("todas", "fracao_ge_0.9_sobre_cobertas"),
               "subida_pontos_sobre_todas": sobe("todas", "fracao_ge_0.9_sobre_todas"),
               "recuperadas_por_estrato": recuperadas, "subida_concentrada_em_escala_grande": concentrada,
               "hipotese_aceita": bool(sobe("todas", "fracao_ge_0.9_sobre_cobertas") >= 10 and concentrada)}
    codigo = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    saida = {"escopo": "só data/cord_sintetico_dev/; val e test do Find it again! não abertos", "k": K, "piso_intensidade": PISO,
             "deslocamentos": list(DS), "escala": "largura da imagem / 1000",
             "criterio_fixado_antes": "aceita se a fração de colagens com correlação >= 0,9 (sobre as cobertas) subir >= 10 pontos entre d=2 e o melhor d e a subida se concentrar em escala >= 1,5 (mais da metade das recuperadas e subida em pontos maior que em escala < 1,5)",
             "tabela": tabela, "decisao": decisao,
             "distribuicao_escala": {"min": float(min(l["escala"] for l in linhas)), "mediana": float(np.median([l["escala"] for l in linhas])), "max": float(max(l["escala"] for l in linhas))},
             "codigo": "notebooks/diagnostico_escala_cord_dev.py", "codigo_sha256": codigo}
    Path("results/diagnostico_escala_cord_dev.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    for nome, t in tabela.items():
        print(f"== {nome}: colagens={t['n_colagens']} cobertas={t['n_cobertas']} ({t['cobertura']:.1%})")
        for d in DS:
            x = t[f"d{d}"]
            print(f"  d={d} mediana={x['mediana']:.3f} >=0,9: {x['n_ge_0.9']}  sobre cobertas {x['fracao_ge_0.9_sobre_cobertas']:.1%}  sobre todas {x['fracao_ge_0.9_sobre_todas']:.1%}")
    print(json.dumps(decisao, ensure_ascii=False, indent=1))
    print("escala:", saida["distribuicao_escala"], "| sha256:", codigo)
