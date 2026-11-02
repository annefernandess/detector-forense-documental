"""Testes com assert para as caixas automáticas. Rodar: python tests/test_regioes.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from regioes import regioes_ela

rng = np.random.default_rng(42)
base = rng.integers(0, 2, (600, 1000)).astype(np.float32)  # ruído de erro baixo (0 ou 1)

# Região de erro conhecida: o bloco forte é encontrado, e a caixa o contém.
com_bloco = base.copy()
com_bloco[200:230, 300:340] = 12
caixas = regioes_ela(com_bloco, 99, 3, contraste_min=2.0)
assert len(caixas) == 1, caixas
(x, y, w, h), intensidade = caixas[0]
assert x <= 300 and y <= 200 and x + w >= 340 and y + h >= 230 and intensidade > 2, caixas[0]

# Sem região forte, nada é destacado (o contraste mínimo impede segmentar ruído).
assert regioes_ela(base, 99, 3, contraste_min=2.0) == []

# Pontos menores que um caractere são descartados.
com_ponto = base.copy()
com_ponto[300:303, 500:503] = 12
assert regioes_ela(com_ponto, 99, 3, contraste_min=2.0) == []

# Caixas com mais de 20% da área da imagem são descartadas.
enorme = base.copy()
enorme[100:450, 100:700] = 12
assert regioes_ela(enorme, 99, 3, contraste_min=2.0) == []

# Ordenação pela intensidade média e limite de caixas.
duas = base.copy()
duas[100:130, 100:140] = 8
duas[400:430, 700:740] = 14
caixas = regioes_ela(duas, 99, 2, contraste_min=2.0)
assert len(caixas) == 2 and caixas[0][1] > caixas[1][1] and caixas[0][0][0] >= 650, caixas
assert len(regioes_ela(duas, 99, 1, contraste_min=2.0)) == 1

# Coordenadas na escala do mapa de entrada: mapa de 2000 px de largura, mesmo bloco ampliado 2x.
grande = np.repeat(np.repeat(com_bloco, 2, axis=0), 2, axis=1)
(x2, y2, w2, h2), _ = regioes_ela(grande, 99, 1, contraste_min=2.0)[0]
assert x2 <= 600 and y2 <= 400 and x2 + w2 >= 680 and y2 + h2 >= 460, (x2, y2, w2, h2)

print("ok")
