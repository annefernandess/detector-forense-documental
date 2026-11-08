"""
Calibração por faixas da pontuação do candidato (v2 + copy_move_par), só com train + val. Regenera as previsões
fora da amostra da validação cruzada 5 x 3 (seed 42) com a receita de congelar_candidato.py (determinística),
confere que reproduzem o limiar congelado e calcula, por faixa de quantis, a fração de adulterados com IC de
Wilson 95%. Sem curva suave, Platt nem isotônica. Não abre o test e não altera o modelo congelado.
Rodar na raiz: python notebooks/calibracao_candidato.py
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
from evaluation import wilson_score_interval, youden_threshold

TMP = Path("/private/tmp/claude-501/-Users-anne-Documents-UFPB-PDI/e66b19d1-edbb-47c8-a1f2-eab9280a9d21/scratchpad")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
NOTA = "tabela estimada sobre previsões fora da amostra de train + val (769 imagens); não é validada no test; a base real de adulterados no conjunto é de cerca de 16%"

if __name__ == "__main__":
    fixada = json.loads(Path("results/calibracao_candidato_fixada.json").read_text(encoding="utf-8"))
    config = json.loads(Path("results/configuracao_candidato.json").read_text(encoding="utf-8"))
    if sha(__file__) != fixada["script_sha256"]:
        sys.exit("o script mudou depois de congelado: parando")
    if sha("results/modelo_candidato.joblib") != config["modelo"]["sha256"]:
        sys.exit("o modelo congelado mudou: parando")

    tabela, colunas = dados()
    X, y = tabela[colunas].to_numpy(), tabela["is_tampered"].to_numpy()
    oof, rotulos = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, y):
        oof.append(modelo().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]); rotulos.append(y[te])
    oof, rotulos = np.concatenate(oof), np.concatenate(rotulos)
    limiar = config["limiar"]["valor"]
    regenerado = youden_threshold(oof[rotulos == 1], oof[rotulos == 0], np.unique(oof))[0]
    assert len(oof) == 2307 and regenerado == limiar, (len(oof), regenerado, limiar)  # as mesmas previsões do limiar

    cortes = np.quantile(oof, [0.2, 0.4, 0.6, 0.8])
    faixa = np.searchsorted(cortes, oof, side="right")  # 0..4; limite inferior inclusivo, superior exclusivo
    inferior = [float(oof.min())] + [float(c) for c in cortes]
    superior = [float(c) for c in cortes] + [float(oof.max())]
    linhas = []
    for f in range(5):
        n, k = int((faixa == f).sum()), int(rotulos[faixa == f].sum())
        lo, hi = wilson_score_interval(k / n, n)
        linhas.append({"faixa": f + 1, "limite_inferior": inferior[f], "limite_superior": superior[f],
                       "n_previsoes": n, "n_adulterados": k, "probabilidade": k / n, "wilson_95": [lo, hi]})
    f_lim = int(np.searchsorted(cortes, limiar, side="right"))
    acima = oof >= limiar
    no_limiar = {"limiar": limiar, "faixa_que_contem_o_limiar": f_lim + 1, "probabilidade_da_faixa": linhas[f_lim]["probabilidade"],
                 "wilson_95_da_faixa": linhas[f_lim]["wilson_95"],
                 "fracao_de_adulterados_entre_previsoes_acima_do_limiar": float(rotulos[acima].mean()),
                 "fracao_de_adulterados_entre_previsoes_abaixo_do_limiar": float(rotulos[~acima].mean())}
    probs = [l["probabilidade"] for l in linhas]
    monotona = bool(all(b > a for a, b in zip(probs, probs[1:])))
    saida = {"congelado_em_script": fixada["congelado_em"], "calculado_em": datetime.now().isoformat(timespec="seconds"),
             "script": "notebooks/calibracao_candidato.py", "script_sha256": fixada["script_sha256"],
             "sha256_modelo_candidato": config["modelo"]["sha256"],
             "n_previsoes": int(len(oof)), "n_imagens": int(len(y)), "base_real_de_adulterados": float(y.mean()),
             "limites_das_faixas": "quantis 0,2; 0,4; 0,6 e 0,8 das previsões fora da amostra", "cortes": [float(c) for c in cortes],
             "faixas": linhas, "no_limiar": no_limiar, "monotonica": monotona,
             "nota": NOTA,
             "ressalva": "a mesma imagem aparece 3 vezes nas previsões (uma por repetição), então as previsões não são independentes e o IC de Wilson é mais estreito do que seria com 769 imagens independentes"}
    print(json.dumps({k: saida[k] for k in ("n_previsoes", "base_real_de_adulterados", "cortes", "monotonica")}, ensure_ascii=False))
    for l in linhas:
        print(f"faixa {l['faixa']}: [{l['limite_inferior']:.4f}; {l['limite_superior']:.4f}] n={l['n_previsoes']} adulterados={l['n_adulterados']} p={l['probabilidade']:.4f} Wilson [{l['wilson_95'][0]:.4f}; {l['wilson_95'][1]:.4f}]")
    print(json.dumps(no_limiar, ensure_ascii=False))
    if not monotona:
        (TMP / "calibracao_nao_monotonica.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
        sys.exit("a probabilidade NÃO cresce com a faixa: parando sem gravar results/calibracao_candidato.json")
    Path("results/calibracao_candidato.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print("gravado results/calibracao_candidato.json | sha256 do script:", fixada["script_sha256"])
