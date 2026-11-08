"""Testes do resíduo de ruído na busca cega de copy-move. Rodar na raiz: python tests/test_copy_move_residuo.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from copy_move_residuo import candidatos, correlacao_residuo, filtrar_pares, residuo

CAIXA_A, CAIXA_B, CAIXA_C = (40, 40, 30, 40), (200, 120, 30, 40), (120, 230, 30, 40)


def _pagina():
    """Página com três 'letras' de mesma forma. B é cópia exata de A (mesmo ruído); C tem a mesma
    forma, mas ruído de digitalização independente."""
    rng = np.random.default_rng(42)
    forma = np.full((40, 30), 225.0)
    forma[6:34, 6:12] = forma[6:34, 18:24] = forma[6:12, 6:24] = 40.0
    pagina = np.full((400, 1000), 225.0)
    letras = {}
    for nome, (x, y, w, h) in zip("ABC", (CAIXA_A, CAIXA_B, CAIXA_C)):
        letras[nome] = forma + rng.normal(0, 6, forma.shape)
        pagina[y:y + h, x:x + w] = letras[nome]
    pagina[CAIXA_B[1]:CAIXA_B[1] + 40, CAIXA_B[0]:CAIXA_B[0] + 30] = letras["A"]  # cópia exata
    return np.clip(np.round(pagina), 0, 255).astype(np.float32)


def test_copia_exata_tem_residuo_quase_1():
    for k in (3, 5):
        assert correlacao_residuo(residuo(_pagina(), k), CAIXA_A, CAIXA_B) > 0.99


def test_trechos_so_com_ruido_independente_tem_correlacao_baixa():
    ruido = np.random.default_rng(42).normal(200, 6, (200, 400)).astype(np.float32)
    assert abs(correlacao_residuo(residuo(ruido, 3), (10, 10, 40, 40), (200, 100, 40, 40))) < 0.3


def test_deslocamento_de_2_px_e_compensado_e_3_px_nao():
    res = residuo(_pagina(), 3)
    assert correlacao_residuo(res, CAIXA_A, (CAIXA_B[0] + 2, CAIXA_B[1] - 2, 30, 40)) > 0.99
    assert correlacao_residuo(res, CAIXA_A, (CAIXA_B[0] + 4, CAIXA_B[1], 30, 40)) < 0.9


def test_trecho_liso_nao_tem_correlacao():
    res = residuo(np.full((100, 100), 200, np.float32), 3)
    assert correlacao_residuo(res, (10, 10, 20, 20), (50, 50, 20, 20)) is None


def test_filtro_do_residuo_mantem_so_a_copia():
    cands = candidatos(_pagina())
    # os três pares de letras de mesma forma passam pelo piso de intensidade
    assert len(filtrar_pares(cands, 0.95)) == 3
    # a forma idêntica já deixa o resíduo do par com ruído independente alto (cerca de 0,87 com k=3,
    # observado ao escrever o teste); por isso o limiar do teste é 0,95
    mantidos = filtrar_pares(cands, 0.95, k=3, limiar=0.95)
    assert len(mantidos) == 1 and mantidos[0][3] > 0.99


def test_par_sem_residuo_definido_e_descartado():
    par = ((0, 0, 4, 4), (10, 10, 4, 4), 0.99, {3: None, 5: 0.9})
    assert filtrar_pares([par], 0.95, k=3, limiar=0.0) == []
    assert len(filtrar_pares([par], 0.95, k=5, limiar=0.5)) == 1


if __name__ == "__main__":
    test_copia_exata_tem_residuo_quase_1()
    test_trechos_so_com_ruido_independente_tem_correlacao_baixa()
    test_deslocamento_de_2_px_e_compensado_e_3_px_nao()
    test_trecho_liso_nao_tem_correlacao()
    test_filtro_do_residuo_mantem_so_a_copia()
    test_par_sem_residuo_definido_e_descartado()
    print("ok")
