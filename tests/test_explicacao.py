"""Testes com assert para a explicação por regras. Rodar: python tests/test_explicacao.py"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explicacao import MUITOS_PARES, explicar, posicao, resumo

TAM = (1000, 1500)
PAR = ((100, 100, 20, 30), (600, 900, 20, 30), 0.995)
PROIBIDAS = ("fraude", "adulterad", "falsific")

# Posições: nove zonas pela posição do centro da caixa.
assert posicao((900, 1400, 50, 50), TAM) == "no canto inferior direito"
assert posicao((10, 10, 50, 50), TAM) == "no canto superior esquerdo"
assert posicao((480, 700, 40, 40), TAM) == "no centro"
assert posicao((960, 1450, 80, 80), TAM) == "no canto inferior direito"  # caixa que passa da borda

casos = {
    # sem sinal algum: a frase fixa pedida
    "vazio": explicar([], TAM, [], True, "Sem sinal relevante"),
    # uma região de ELA, sem repetição
    "regiao": explicar([((900, 1400, 50, 50), 3.1)], TAM, [], True, "Revisar"),
    # um par de copy-move, sem região
    "par": explicar([], TAM, [PAR], True, "Sem sinal relevante"),
    # duas regiões e dois pares
    "ambos": explicar([((500, 700, 40, 40), 3.4), ((10, 10, 40, 40), 2.9)], TAM, [PAR, PAR], True, "Revisar"),
    # muitos pares: texto repetitivo
    "muitos": explicar([], TAM, [PAR] * MUITOS_PARES, True, "Sem sinal relevante"),
    # sem histórico JPEG
    "sem_historico": explicar([], TAM, [], False, "Sem sinal relevante"),
    # classificador pede revisão, mas nenhuma região foi destacada
    "revisar_sem_sinal": explicar([], TAM, [], True, "Revisar"),
}
assert casos["vazio"] == "Nenhuma região com diferença de compressão acima do esperado foi encontrada."
assert "canto inferior direito" in casos["regiao"] and "pode indicar edição" in casos["regiao"]
assert casos["regiao"].endswith("não foi encontrada repetição de trecho.")
assert "um possível trecho repetido entre duas regiões" in casos["par"] and "conferida visualmente" in casos["par"]
assert casos["par"].startswith("Nenhuma região")
assert "2 regiões" in casos["ambos"] and "no centro" in casos["ambos"] and "2 possíveis trechos repetidos" in casos["ambos"]
assert "comuns em texto repetitivo" in casos["muitos"] and f"{MUITOS_PARES} pares" in casos["muitos"]
assert "JPEG Ghost não se aplica" in casos["sem_historico"] and casos["sem_historico"].count(". ") == 1
assert "indica revisão" in casos["revisar_sem_sinal"]
assert "JPEG Ghost" not in casos["vazio"]  # com histórico, o Ghost não aparece no texto

# Nunca afirma alteração, e tem no máximo duas frases.
for nome, texto in casos.items():
    assert not any(p in texto.lower() for p in PROIBIDAS), (nome, texto)
    assert texto.count(".") <= 2 and texto.endswith("."), (nome, texto)

# O limiar do texto confere com o registrado na calibração.
config = ROOT / "results/regioes_config.json"
if config.exists():
    assert json.loads(config.read_text(encoding="utf-8"))["explicacao"]["limite_muitos_pares"] == MUITOS_PARES

# Resumo em linguagem simples: quatro situações, sem jargão e sem afirmar alteração.
resumos = {(f, n): resumo(f, n) for f in ("Revisar", "Sem sinal relevante") for n in (0, 3)}
assert "revisão humana" in resumos[("Revisar", 0)] and "trechos repetidos" not in resumos[("Revisar", 0)]
assert "revisão humana" in resumos[("Revisar", 3)] and "possíveis trechos repetidos" in resumos[("Revisar", 3)]
assert "não tem sinais fortes" in resumos[("Sem sinal relevante", 0)]
assert "possíveis trechos repetidos" in resumos[("Sem sinal relevante", 3)]
for texto in resumos.values():
    assert not any(p in texto.lower() for p in PROIBIDAS), texto
    assert not re.search(r"\b(ela|jpeg|ghost|copy-move|auc)\b", texto.lower()), texto  # sem jargão no resumo
    assert texto.count(".") <= 3 and texto.endswith("."), texto

print("ok")
