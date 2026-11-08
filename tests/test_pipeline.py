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
from copy_move_feature import feature_copy_move
from dataset_loader import load_gray
from pipeline import (analisar_imagem, carregar_candidato, faixa_e_posicao, metricas_candidato, pontuacoes_referencia,
                      verificar_hash)
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

# O hash do modelo é conferido antes do uso: hash errado para a execução; o certo passa.
try:
    verificar_hash(b"conteudo qualquer", "0" * 64, "modelo de teste")
    raise AssertionError("esperava RuntimeError")
except RuntimeError:
    pass
verificar_hash(b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc")
modelo, config = carregar_candidato()
assert len(config["features"]) == 13 and config["features"][-1] == "copy_move_par"
assert config["limiar"]["valor"] == 0.38857134275922206
m = metricas_candidato()
assert (m["sensibilidade"], m["especificidade"]) == (config["limiar"]["sensibilidade"], config["limiar"]["especificidade"])

# A referência do percentil: 2307 previsões fora da amostra, com o hash conferido na carga.
referencia = pontuacoes_referencia()
assert len(referencia) == 2307 and 0 <= referencia.min() and referencia.max() <= 1
assert faixa_e_posicao(config["limiar"]["valor"], config["limiar"]["valor"], referencia)[0] == "Revisar"

# Consistência com o lote: a análise avulsa dá a mesma pontuação e a mesma feature de copy-move que o lote
# (CSVs de features de train e val + `copy_move_par`), para 5 imagens do val.
val_dir = ROOT / "data/findit2/val"
caches = [ROOT / "data/features_val.csv", ROOT / "data/features_v2_val.csv", ROOT / "data/features_copy_move_cv.csv"]
if not (val_dir.exists() and all(c.exists() for c in caches)):
    print("ok (parcial: dataset ou features do lote ausentes, consistência com o lote não testada)")
    sys.exit(0)
lote = (pd.read_csv(caches[0]).merge(pd.read_csv(caches[1]), on="image")
        .merge(pd.read_csv(caches[2])[["image", "copy_move_par"]], on="image").set_index("image"))
for nome in random.Random(42).sample(sorted(lote.index.intersection([p.name for p in val_dir.glob("*.png")])), 5):
    r = analisar_imagem(val_dir / nome)
    esperado = modelo.predict_proba(lote.loc[[nome], config["features"]].to_numpy())[0, 1]
    diff = abs(r["pontuacao"] - esperado)
    print(f"{nome}: diferença = {diff:.3e} | faixa = {r['faixa']} | copy_move_par = {r['features']['copy_move_par']}")
    assert diff < 1e-9, (nome, diff)
    assert r["features"]["copy_move_par"] == lote.loc[nome, "copy_move_par"] == int(len(r["copy_move_pares"]) > 0)
    assert r["features"]["copy_move_par"] == feature_copy_move(load_gray(val_dir / nome), **{
        k: config["configuracao_tecnica_1"][c] for k, c in (("piso", "piso_intensidade"), ("k", "k"), ("limiar", "limiar_residuo"))})
    assert r["limiar"] == config["limiar"]["valor"] and r["posicao"] is not None
    assert isinstance(r["regioes"], list) and len(r["regioes"]) <= 2 and isinstance(r["explicacao"], str)
    assert not any(p in r["explicacao"].lower() for p in ("fraude", "adulterad", "falsific")), r["explicacao"]
    assert list(r["features"]) == config["features"]

print("ok")
