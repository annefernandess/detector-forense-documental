"""
Limite de "muitos pares" do texto explicativo (src/explicacao.py), para a técnica 1 (copy_move_residuo com a
configuração congelada em results/configuracao_candidato.json). Regra, fixada antes de calcular: o limite é o
percentil 99 do número de pares aprovados por imagem nas autênticas de train + val (642 imagens), arredondado para
cima; "muitos" = esse número de pares ou mais. Sem test e sem o v2. Atualiza só o bloco "explicacao" de
results/regioes_config.json. Sequencial (sem fork). Rodar na raiz do projeto: python app/calibrar_limite_pares.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from copy_move_residuo import candidatos, filtrar_pares
from dataset_loader import load_dataset, load_gray

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

if __name__ == "__main__":
    config = json.loads((ROOT / "results/configuracao_candidato.json").read_text(encoding="utf-8"))
    t = config["configuracao_tecnica_1"]
    autenticas = [s.image_path for split in ("train", "val") for s in load_dataset(str(ROOT / "data/findit2"), split)
                  if s.image_path.exists() and not s.is_tampered]
    assert len(autenticas) == 642, len(autenticas)
    pares = np.array([len(filtrar_pares(candidatos(load_gray(c), ks=(t["k"],)), t["piso_intensidade"], t["k"],
                                        t["limiar_residuo"])) for c in autenticas])
    distribuicao = {"imagens": int(len(pares)), "minimo": int(pares.min()), "mediana": float(np.median(pares)),
                    "p90": float(np.percentile(pares, 90)), "p95": float(np.percentile(pares, 95)),
                    "p99": float(np.percentile(pares, 99)), "maximo": int(pares.max()),
                    "imagens_com_algum_par": int((pares > 0).sum())}
    limite = int(np.ceil(distribuicao["p99"]))
    caminho = ROOT / "results/regioes_config.json"
    regioes = json.loads(caminho.read_text(encoding="utf-8"))
    regioes["explicacao"] = {
        "tecnica_pares": {"piso_intensidade": t["piso_intensidade"], "k": t["k"], "limiar_residuo": t["limiar_residuo"],
                          "fonte": "results/configuracao_candidato.json"},
        "limite_muitos_pares": limite,
        "origem_limite": "percentil 99 do número de pares aprovados pela técnica 1 por imagem, nas autênticas de train + val "
                         "(642 imagens), arredondado para cima; 'muitos' = esse número de pares ou mais",
        "distribuicao_pares_autenticas": distribuicao,
        "calculado_em": datetime.now().isoformat(timespec="seconds")}
    caminho.write_text(json.dumps(regioes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(distribuicao, ensure_ascii=False))
    print("limite_muitos_pares =", limite)
