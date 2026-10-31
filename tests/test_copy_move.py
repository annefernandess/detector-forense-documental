"""Testes com assert para o copy-move e o limiar de Youden. Rodar: python tests/test_copy_move.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from copy_move import correlation_peak
from evaluation import youden_threshold

rng = np.random.default_rng(42)
img = rng.integers(0, 256, (100, 120)).astype(np.float32)
img[60:70, 80:95] = img[10:20, 5:20]  # cópia de um trecho para outro ponto

# A cópia encontra a origem com correlação 1, e não a si mesma.
peak, loc = correlation_peak(img, (80, 60, 15, 10))
assert peak > 0.999 and loc == (5, 10, 15, 10), (peak, loc)

# Correlação é invariante a ganho e deslocamento de brilho (slide 270).
img2 = img.copy()
img2[60:70, 80:95] = 2 * img[10:20, 5:20] + 7
assert correlation_peak(img2, (80, 60, 15, 10))[0] > 0.999

# Região lisa: correlação indefinida.
img3 = img.copy()
img3[0:10, 0:10] = 255
assert correlation_peak(img3, (0, 0, 10, 10)) == (None, None)

# Youden: J empata em 0,667 de 0,95 a 0,999; fica o primeiro candidato.
t, tpr, fpr = youden_threshold([0.99, 1.0, 1.0], [0.5, 0.9, 0.995], [0.9, 0.95, 0.99, 0.999])
assert (t, tpr, round(fpr, 3)) == (0.95, 1.0, 0.333), (t, tpr, fpr)

print("ok")
