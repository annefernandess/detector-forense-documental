"""Testes da lógica de escolha do ajuste. Rodar na raiz: python tests/test_ajuste_parametros.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ajuste_parametros import PADROES, configuracoes, escolher


def _linha(b, dif, rmin, amax):
    return {"block_size": b, "diferenca": dif, "real_ic_min": rmin, "acaso_ic_max": amax}


def test_grade():
    c = configuracoes()
    assert len(c) == 5 + 3 + 6 and sum(t == "ela" for t, _ in c) == 5
    assert [p for t, p in c if t == "ghost"][0] == PADROES["ghost"]


def test_escolher():
    padrao = {"block_size": 16}
    ls = [_linha(8, 0.30, 0.2, 0.1), _linha(16, 0.20, 0.1, 0.1), _linha(32, 0.26, 0.2, 0.1)]
    assert escolher(ls, padrao)[0]["block_size"] == 8                       # ganha 10 pontos
    ls[0]["real_ic_min"] = 0.05                                             # IC toca o acaso
    assert escolher(ls, padrao)[0]["block_size"] == 32                      # ganha 6 pontos
    ls[2]["diferenca"] = 0.24                                               # ganha só 4 pontos
    assert escolher(ls, padrao) == (ls[1], False)


if __name__ == "__main__":
    test_grade(); test_escolher(); print("ok")
