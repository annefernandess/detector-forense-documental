"""Testes do ruído local revisado. Rodar na raiz: python tests/test_ruido_local.py"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

from ruido_fundo import score_regiao
from ruido_local import calibrar, fracao_suspeita, mapa_z, valores_ruido_local

B = 32


def _papel(sigma, seed=42):
    """Papel 512 x 512 com ruído gaussiano (sigma 0 = fundo plano) e linhas escuras de texto para o Otsu."""
    img = np.full((512, 512), 220.0)
    if sigma:
        img += np.random.default_rng(seed).normal(0, sigma, img.shape)
    for y in (20, 100, 300, 460):
        img[y:y + 3, 30:200] = 40
    return np.clip(np.round(img), 0, 255).astype(np.float32)


def _calibracao():
    """Calibração com 'autênticas' sintéticas como as reais: a maioria de papel plano, algumas com ruído fraco."""
    return calibrar([valores_ruido_local(_papel(s, seed=i), B) for i, s in enumerate((0, 0, 0, 0, 0.6, 1.0, 1.4))])


def test_bloco_com_desvio_diferente_tem_z_alto():
    media, desvio = _calibracao()
    img = _papel(1.4)
    img[128:192, 192:256] += np.random.default_rng(7).normal(0, 12, (64, 64)).astype(np.float32)
    z = mapa_z(valores_ruido_local(img, B), media, desvio)
    assert score_regiao(z, (192, 128, 64, 64), B) > 3  # acima do limiar de bloco suspeito (|z| > 3)
    assert score_regiao(z, (352, 352, 64, 64), B) < 3
    assert fracao_suspeita(z) > 0


def test_fundo_plano_nao_tem_nan_nem_blocos_suspeitos():
    media, desvio = _calibracao()
    valores = valores_ruido_local(_papel(0), B)
    assert np.isfinite(valores[~np.isnan(valores)]).all()
    assert (valores[~np.isnan(valores)] == np.float32(np.log(0.1))).all()  # piso: log(0 + 0,1)
    z = mapa_z(valores, media, desvio)
    assert np.isfinite(z[~np.isnan(z)]).all() and fracao_suspeita(z) == 0.0


def test_blocos_com_texto_sao_ignorados_e_imagem_sem_bloco_valido_da_none():
    img = _papel(1.4)
    img[64:96, 64:96] = 40  # bloco inteiro de texto: sem fundo
    valores = valores_ruido_local(img, B)
    assert np.isnan(valores[2, 2]) and (~np.isnan(valores)).sum() > 100
    assert valores_ruido_local(np.full((16, 16), 200, np.float32), B) is None  # sem bloco inteiro
    assert fracao_suspeita(None) is None


def test_calibracao_tem_media_e_desvio_dos_valores():
    m1 = np.array([[1.0, 3.0], [np.nan, 5.0]], np.float32)
    media, desvio = calibrar([m1, None])
    assert media == 3.0 and abs(desvio - np.sqrt(8 / 3)) < 1e-6


if __name__ == "__main__":
    test_bloco_com_desvio_diferente_tem_z_alto()
    test_fundo_plano_nao_tem_nan_nem_blocos_suspeitos()
    test_blocos_com_texto_sao_ignorados_e_imagem_sem_bloco_valido_da_none()
    test_calibracao_tem_media_e_desvio_dos_valores()
    print("ok")
