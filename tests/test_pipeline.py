"""Testes com assert para o pipeline da interface. Rodar: python tests/test_pipeline.py"""

import json
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from overlays import painel_copy_move, painel_ela, painel_ghost, painel_original, recortar
from pipeline import METRICAS_TEST, analisar_imagem, faixa_e_posicao
import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

# Faixas: o limiar é "Revisar" (igual ou acima); a posição conta só as pontuações menores.
ref = np.array([0.1, 0.2, 0.3, 0.4])
assert faixa_e_posicao(0.5, 0.5, ref) == ("Revisar", 100.0)
assert faixa_e_posicao(0.25, 0.5, ref) == ("Sem sinal relevante", 50.0)
assert faixa_e_posicao(0.25, 0.5, None) == ("Sem sinal relevante", None)


# Painéis: o formato da saída é o da imagem, e opacidade 0 devolve a imagem original.
cinza = (np.arange(60 * 80).reshape(60, 80) % 256).astype(np.uint8)
original = painel_original(cinza)
ela = (np.arange(60 * 80).reshape(60, 80) % 256).astype(np.uint8)
assert painel_ela(cinza, ela, 0.0).shape == (60, 80, 3) and (painel_ela(cinza, ela, 0.0) == original).all()
assert not (painel_ela(cinza, ela, 1.0) == original).all()
gm = np.array([[0, 8], [-1, 4]], np.int32)  # 2x2 blocos de 16 px, escala 1,25
g = painel_ghost(cinza, gm, 1.25, 1.0, tuple(range(50, 95, 5)))
assert g.shape == (60, 80, 3)
assert not (g[5:15, 25:35] == original[5:15, 25:35]).all()  # bloco (0, 1), valor 8: colorido
assert not (g[25:35, 25:35] == original[25:35, 25:35]).all()  # bloco (1, 1), valor 4: colorido
assert (g[30:35, 2:8] == original[30:35, 2:8]).all()  # bloco liso (-1): sem cor
par = [((5, 5, 10, 10), (50, 30, 10, 10), 0.995)]
c0 = painel_copy_move(cinza, par, 1.0, 0.0)
assert (c0 == original).all() and not (painel_copy_move(cinza, par, 1.0, 1.0) == original).all()
assert (painel_copy_move(cinza, [], 1.0, 1.0) == original).all()  # sem pares: imagem intacta
assert recortar(original, 0.25, 0.5, 0.5, 0.25).shape == (15, 40, 3)

# Os números fixos da interface conferem com as saídas gravadas do notebook 04.
nb = json.loads((ROOT / "notebooks/04_classificador.ipynb").read_text(encoding="utf-8"))
saida = "".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code"
                for o in c["outputs"] if o.get("output_type") == "stream")
for trecho in ("precisão: 20.0%", "sensibilidade: 45.7%", "especificidade: 65.0%", "AUC: 0.626",
               "(IC 95%: 30.5% a 61.8%)", "(IC 95%: 12.7% a 30.0%)", "acurácia = 83.9%"):
    assert trecho in saida, trecho
m = METRICAS_TEST
assert (m["marcadas_adulteradas"], m["marcadas_autenticas"]) == (16, 64)
assert round(16 / 35, 3) == m["sensibilidade"] and round(16 / 80, 3) == m["precisao"]
assert round(119 / 183, 3) == m["especificidade"] and round(183 / 218, 3) == m["acuracia_tudo_autentico"]

# Consistência com o lote: a análise avulsa dá a probabilidade gravada para 5 imagens do val.
csv = ROOT / "results/probabilidades_val_v2.csv"
val_dir = ROOT / "data/findit2/val"
if not (csv.exists() and val_dir.exists()):
    print("ok (parcial: dataset ou probabilidades do val ausentes, consistência com o lote não testada)")
    sys.exit(0)
ref = pd.read_csv(csv).set_index("image")["prob_v2"]
for nome in random.Random(42).sample(sorted(ref.index), 5):
    r = analisar_imagem(val_dir / nome)
    diff = abs(r["pontuacao"] - ref[nome])
    print(f"{nome}: diferença = {diff:.3e} | faixa = {r['faixa']}")
    assert diff < 1e-9, (nome, diff)
    assert isinstance(r["regioes"], list) and len(r["regioes"]) <= 2 and isinstance(r["explicacao"], str)
    assert not any(p in r["explicacao"].lower() for p in ("fraude", "adulterad", "falsific")), r["explicacao"]
    assert list(r["features"]) == json.loads((ROOT / "results/configuracao_congelada.json").read_text(encoding="utf-8"))["features"]

print("ok")
