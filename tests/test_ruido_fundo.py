"""Testes dos mapas de ruído local e de fundo do papel. Rodar na raiz: python tests/test_ruido_fundo.py"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

from comparacao_cv import diferenca_pareada
from copy_move_residuo import residuo
from ruido_fundo import (features_mapa, mapa_fundo_papel, mapa_ruido_local, mascara_fundo, score_regiao,
                         superficie_polinomial)

B = 32


def _papel(sigma=2.0, base=220.0):
    """Papel 512 x 512 com ruído gaussiano e algumas linhas escuras (texto), para o Otsu achar o texto."""
    rng = np.random.default_rng(42)
    img = base + rng.normal(0, sigma, (512, 512))
    for y in (20, 100, 300, 460):
        img[y:y + 3, 30:200] = 40
    return img.astype(np.float32)


def test_bloco_com_ruido_diferente_aparece_com_z_alto():
    img = _papel()
    rng = np.random.default_rng(7)
    img[128:192, 192:256] += rng.normal(0, 12, (64, 64)).astype(np.float32)
    mapa = mapa_ruido_local(residuo(img, 3), mascara_fundo(img), B)
    assert score_regiao(mapa, (192, 128, 64, 64), B) > 5
    assert score_regiao(mapa, (352, 352, 64, 64), B) < 3
    p99, fracao = features_mapa(mapa)
    assert p99 > 3 and fracao > 0


def test_imagem_uniforme_nao_tem_blocos_suspeitos():
    img = _papel()
    mapa = mapa_ruido_local(residuo(img, 3), mascara_fundo(img), B)
    p99, fracao = features_mapa(mapa)
    assert mapa is not None and fracao <= 0.02 and p99 < 3.5, (p99, fracao)


def test_fundo_do_papel_acha_bloco_com_deslocamento_de_brilho():
    y, x = np.mgrid[0:512, 0:512] / 511.0
    iluminacao = 200 + 20 * x + 10 * y - 15 * x * x + 8 * x * y
    img = (iluminacao + np.random.default_rng(42).normal(0, 1, (512, 512))).astype(np.float32)
    for linha in (20, 100, 300, 460):
        img[linha:linha + 3, 30:200] = 40
    limpo = mapa_fundo_papel(img, mascara_fundo(img), B)
    assert features_mapa(limpo)[1] <= 0.02
    img[128:192, 192:256] += 12
    mapa = mapa_fundo_papel(img, mascara_fundo(img), B)
    assert score_regiao(mapa, (192, 128, 64, 64), B) > 5


def test_superficie_recupera_polinomio_de_grau_2():
    y, x = np.mgrid[0:200, 0:300]
    u, v = x / 299 * 2 - 1, y / 199 * 2 - 1
    img = (150 + 30 * u - 20 * v + 10 * u * u + 5 * u * v - 8 * v * v).astype(np.float32)
    superficie = superficie_polinomial(img, np.ones(img.shape, bool))
    assert np.abs(superficie - img).max() < 0.05


def test_mapa_indefinido_com_poucos_blocos_e_regiao_sem_bloco_valido():
    pequena = np.full((64, 64), 200, np.float32)
    assert mapa_ruido_local(pequena, np.ones((64, 64), bool), B) is None  # 4 blocos < 10
    mapa = np.full((16, 16), np.nan, np.float32)
    mapa[0, 0] = 4.0
    assert score_regiao(mapa, (0, 0, 10, 10), B) == 4.0
    assert score_regiao(mapa, (200, 200, 10, 10), B) is None  # só blocos inválidos
    assert score_regiao(mapa, (5000, 5000, 10, 10), B) is None  # fora do mapa


def test_diferenca_pareada():
    a = np.array([0.7, 0.72, 0.68, 0.71, 0.69, 0.73, 0.7, 0.7, 0.71, 0.69, 0.72, 0.68, 0.7, 0.71, 0.69])
    zero = diferenca_pareada(a, a)
    assert zero["diferenca_media"] == 0 and zero["ic95_corrigido"] == [0.0, 0.0]
    d = diferenca_pareada(a + 0.05 + np.linspace(-0.01, 0.01, 15), a)
    assert abs(d["diferenca_media"] - 0.05) < 1e-9 and d["ic95_corrigido"][0] > 0
    assert d["ic95_corrigido"][1] - d["ic95_corrigido"][0] > d["ic95_ingenuo"][1] - d["ic95_ingenuo"][0]


if __name__ == "__main__":
    test_bloco_com_ruido_diferente_aparece_com_z_alto()
    test_imagem_uniforme_nao_tem_blocos_suspeitos()
    test_fundo_do_papel_acha_bloco_com_deslocamento_de_brilho()
    test_superficie_recupera_polinomio_de_grau_2()
    test_mapa_indefinido_com_poucos_blocos_e_regiao_sem_bloco_valido()
    test_diferenca_pareada()
    print("ok")
