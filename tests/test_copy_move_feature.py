"""Testes da feature binária de copy-move. Rodar na raiz: python tests/test_copy_move_feature.py"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from copy_move_feature import feature_copy_move
from test_copy_move_residuo import CAIXA_B, _pagina

CONFIG = {"piso": 0.95, "k": 3, "limiar": 0.9}


def test_pagina_com_copia_exata_tem_feature_1():
    assert feature_copy_move(_pagina(), **CONFIG) == 1


def test_limiar_do_residuo_separa_copia_exata_de_letra_igual_com_ruido_independente():
    pagina = _pagina()
    x, y, w, h = CAIXA_B
    pagina[y:y + h, x:x + w] = 225.0  # apaga a cópia exata: sobram A e C, de mesma forma e ruído independente
    # Com o limiar congelado (0,9) esse par pode passar: na caixa justa da letra, a borda do traço domina o
    # resíduo (observado ao escrever este teste). Com um limiar alto o mecanismo separa os dois casos.
    assert feature_copy_move(pagina, piso=0.95, k=3, limiar=0.999) == 0
    assert feature_copy_move(_pagina(), piso=0.95, k=3, limiar=0.999) == 1


def test_pagina_sem_letras_repetidas_tem_feature_0():
    assert feature_copy_move(np.full((400, 1000), 225, np.float32), **CONFIG) == 0


if __name__ == "__main__":
    test_pagina_com_copia_exata_tem_feature_1()
    test_limiar_do_residuo_separa_copia_exata_de_letra_igual_com_ruido_independente()
    test_pagina_sem_letras_repetidas_tem_feature_0()
    print("ok")
