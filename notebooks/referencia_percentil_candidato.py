"""
Referência do percentil da interface: previsões fora da amostra do candidato (v2 + copy_move_par) na validação
cruzada 5 x 3 (seed 42), nas 769 imagens de train + val, 2307 previsões (3 por imagem, agrupadas). Regenera as
previsões com a receita de congelar_candidato.py (determinística), confere que reproduzem o limiar congelado e grava
as pontuações ordenadas, sem nomes de imagem, em results/referencia_percentil_candidato.json. Não abre o test.
Rodar na raiz: python notebooks/referencia_percentil_candidato.py
"""
import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "notebooks")
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from congelar_candidato import dados, modelo
from evaluation import youden_threshold

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

if __name__ == "__main__":
    destino = Path("results/referencia_percentil_candidato.json")
    assert not destino.exists(), "a referência já existe"
    config = json.loads(Path("results/configuracao_candidato.json").read_text(encoding="utf-8"))
    assert sha("results/modelo_candidato.joblib") == config["modelo"]["sha256"], "o modelo congelado mudou"
    tabela, colunas = dados()
    X, y = tabela[colunas].to_numpy(), tabela["is_tampered"].to_numpy()
    oof, rotulos = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, y):
        oof.append(modelo().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]); rotulos.append(y[te])
    oof, rotulos = np.concatenate(oof), np.concatenate(rotulos)
    limiar = youden_threshold(oof[rotulos == 1], oof[rotulos == 0], np.unique(oof))[0]
    assert len(oof) == 2307 and limiar == config["limiar"]["valor"], (len(oof), limiar)  # as mesmas previsões do limiar
    previsoes = np.sort(oof).astype(np.float64)
    saida = {"origem": "previsões fora da amostra da validação cruzada 5 x 3 (seed 42) do candidato nas 769 imagens de train + val, 3 por imagem, agrupadas; sem test e sem o v2",
             "gerado_em": datetime.now().isoformat(timespec="seconds"), "n_previsoes": int(len(previsoes)), "n_imagens": int(len(y)),
             "limiar_reproduzido": float(limiar), "sha256_modelo_candidato": config["modelo"]["sha256"],
             "script": "notebooks/referencia_percentil_candidato.py", "sha256_script": sha(__file__),
             "sha256_previsoes": hashlib.sha256(previsoes.tobytes()).hexdigest(),
             "formato_do_hash": "SHA-256 dos bytes de np.array(previsoes, dtype=float64), na ordem do arquivo (ordenadas)",
             "previsoes": [float(v) for v in previsoes]}
    destino.write_text(json.dumps(saida, ensure_ascii=False), encoding="utf-8")
    print("n:", saida["n_previsoes"], "| limiar reproduzido:", limiar, "| sha256 das previsões:", saida["sha256_previsoes"])
    print("sha256 do arquivo:", sha(destino))
