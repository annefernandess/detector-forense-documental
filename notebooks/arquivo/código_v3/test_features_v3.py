"""Testes com assert para o alinhamento de linha de base. Rodar: python tests/test_features_v3.py"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # notebooks/arquivo/código_v3
sys.path.insert(0, str(HERE.parents[2] / "src"))      # src/ na raiz do projeto
sys.path.insert(0, str(HERE))                          # features_v3.py, nesta pasta

import numpy as np

from features_v3 import baseline_deviations

# Uma linha com 12 "caracteres" iguais, já em largura 1000; o oitavo desce 6 px.
# Altura de 1000 px: componentes acima de 5% da altura da imagem são descartados.
img = np.full((1000, 1000), 255, np.float32)
for i in range(12):
    x0 = 60 + i * 70
    dy = 6 if i == 7 else 0
    img[80 + dy:106 + dy, x0:x0 + 16] = 0
dev_px, dev_norm = baseline_deviations(img)
assert len(dev_px) == 12, dev_px
# a ordem dos componentes segue a varredura do OpenCV, por isso se testa o valor, não a posição
assert np.isclose(np.abs(dev_px).max(), 6.0), dev_px
assert np.sum(np.abs(dev_px) > 1) == 1, dev_px  # só o caractere deslocado passa de 1 px

print("ok")
