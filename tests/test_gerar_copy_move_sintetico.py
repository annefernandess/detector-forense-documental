"""Testes do gerador de copy-move sintético. Rodar na raiz: python tests/test_gerar_copy_move_sintetico.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gerar_copy_move_sintetico import SEED, colar_caixa_texto, mascara_texto


def _pagina():
    img = np.full((400, 600), 255, np.uint8)
    for k, (x, y) in enumerate([(50, 50), (200, 60), (350, 200), (100, 300), (450, 320)]):
        img[y:y + 12, x:x + 10 + k] = 0  # blocos escuros de formas diferentes
    return img


def test_cola_sem_transformacao_e_sem_sobreposicao():
    img = _pagina()
    editada, (x, y, w, h) = colar_caixa_texto(img, np.random.default_rng(SEED))
    diff = editada != img
    assert diff.any()
    ys, xs = np.nonzero(diff)
    assert xs.min() >= x and xs.max() < x + w and ys.min() >= y and ys.max() < y + h
    # o bloco colado é cópia exata de algum trecho original de mesmo tamanho
    colado = editada[y:y + h, x:x + w]
    assert any(np.array_equal(colado, img[j:j + h, i:i + w])
               for j in range(img.shape[0] - h) for i in range(img.shape[1] - w)
               if not (abs(i - x) < w and abs(j - y) < h))


def test_destino_tem_texto_e_e_deterministico():
    img = _pagina()
    a = colar_caixa_texto(img, np.random.default_rng(SEED))
    b = colar_caixa_texto(img, np.random.default_rng(SEED))
    assert a[1] == b[1] and np.array_equal(a[0], b[0])
    x, y, w, h = a[1]
    assert mascara_texto(img)[y:y + h, x:x + w].any()


def test_pagina_vazia_levanta():
    try:
        colar_caixa_texto(np.full((100, 100), 255, np.uint8), np.random.default_rng(SEED))
    except ValueError:
        return
    raise AssertionError("esperava ValueError")


if __name__ == "__main__":
    test_cola_sem_transformacao_e_sem_sobreposicao()
    test_destino_tem_texto_e_e_deterministico()
    test_pagina_vazia_levanta()
    print("ok")
