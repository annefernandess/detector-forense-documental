"""Testes de linguagem da interface. Rodar: python tests/test_app_textos.py"""

import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import textos as T
from pipeline import metricas_candidato

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

JARGAO_LIVRE = re.compile(r"JPEG Ghost|copy-move|z-score|colormap", re.I)
JARGAO_EXATO = re.compile(r"\bELA\b|\bAUC\b|\bIoU\b|\bp95\b")  # caixa exata: "Ela" é pronome


def tem_jargao(texto):
    return JARGAO_LIVRE.search(texto) or JARGAO_EXATO.search(texto)


PROIBIDAS = re.compile(r"adulterad|falsific", re.I)


# O detector de jargão reprova termos técnicos, mas não o pronome "Ela" nem a palavra "ela".
assert tem_jargao("O mapa ELA mostra") and tem_jargao("A técnica JPEG Ghost") and tem_jargao("busca copy-move")
assert tem_jargao("um z-score alto") and tem_jargao("a AUC é 0,6") and tem_jargao("o colormap inferno")
assert not tem_jargao("Ela não decide. Ela sinalizou 1 em cada 2.") and not tem_jargao("histórico JPEG detectado")
assert PROIBIDAS.search("documento adulterado") and PROIBIDAS.search("imagem falsificada")
assert not PROIBIDAS.search("possível edição")


def textos_do_modulo():
    """(nome, texto) de todo texto de T, com a marca de ajuda para os textos de ajuda."""
    for nome in dir(T):
        valor = getattr(T, nome)
        if nome.startswith("_") or nome == "JARGAO":
            continue
        if isinstance(valor, str):
            yield nome, valor, nome == "POSICAO_AJUDA"
        elif isinstance(valor, list):
            for i, item in enumerate(valor):
                if isinstance(item, str):
                    yield f"{nome}[{i}]", item, False
                elif isinstance(item, tuple):
                    for j, parte in enumerate(item):
                        yield f"{nome}[{i}][{j}]", parte, False
        elif isinstance(valor, dict) and nome == "MAPAS":
            for chave, campos in valor.items():
                for campo, texto in campos.items():
                    yield f"MAPAS.{chave}.{campo}", texto, campo == "ajuda"
        elif isinstance(valor, dict) and nome == "FAIXAS":
            for chave, (rotulo, icone) in valor.items():
                yield f"FAIXAS.{chave}", rotulo, False


def frases(texto):
    return [f for f in re.split(r"(?<=[.!?])\s+", texto.strip()) if f]


# 1. Nos textos do módulo: sem palavras proibidas; jargão só nos textos de ajuda, que têm 2 frases.
n_ajuda = 0
for nome, texto, eh_ajuda in textos_do_modulo():
    assert not PROIBIDAS.search(texto), (nome, texto)
    if eh_ajuda:
        n_ajuda += 1
        assert len(frases(texto)) == 2, (nome, texto)
    else:
        assert not tem_jargao(texto), (nome, texto)
assert n_ajuda == 4, n_ajuda  # posição + 3 técnicas

# "fraude" só aparece negada, em frases fixas.
for nome, texto, _ in textos_do_modulo():
    if "fraude" in texto.lower():
        assert re.search(r"não (confirma|decide)", texto, re.I), (nome, texto)

# 2. Números: vêm de results/configuracao_candidato.json (sensibilidade 75,3% e especificidade 60,1% nas previsões
# fora da amostra de train + val), conferidos com as frases; a declaração e a posição seguem os textos combinados.
m = metricas_candidato()
assert (round(m["sensibilidade"], 3), round(m["especificidade"], 3)) == (0.753, 0.601), m
medidos = T.numeros_medidos(m)
for trecho in ("cerca de 75% dos documentos editados", "cerca de 40% dos documentos sem edição", "otimistas"):
    assert trecho in medidos, trecho
assert T.posicao_frase(72.4) == "Esta imagem está em posição mais alta que 72% dos recibos de referência deste tipo."
assert T.DECLARACAO == ("Esta ferramenta foi validada apenas para recibos semelhantes aos do conjunto Find it again!. "
                        "Ela ordena documentos para revisão e não confirma fraude.")

# 3. Página renderizada: sem jargão fora dos textos de ajuda, nos três estados.
from streamlit.testing.v1 import AppTest


def visiveis(at):
    """Textos exibidos fora dos expansíveis de ajuda."""
    ajuda = {m.value for e in at.expander for m in e.markdown}
    partes = [e.value for tipo in (at.markdown, at.caption, at.header, at.subheader, at.title, at.info, at.error)
              for e in tipo if e.value not in ajuda and not e.value.lstrip().startswith("<style>")]
    return "\n".join(partes), ajuda


estados = {}
at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=300)
at.run()
estados["inicial"] = (at, None)
if len(at.button):
    at.button[0].click().run()
    estados["exemplo"] = (at, None)
    at2 = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=300)
    at2.query_params["demo"] = "1"
    at2.query_params["exemplo"] = at.session_state["exemplo"]
    at2.run()
    estados["demonstração"] = (at2, None)

for nome, (app, _) in estados.items():
    assert not app.exception, (nome, [e.value for e in app.exception])
    texto, ajuda = visiveis(app)
    assert not PROIBIDAS.search(texto), (nome, PROIBIDAS.search(texto).group())
    assert not tem_jargao(texto), (nome, tem_jargao(texto).group())
    assert T.DECLARACAO in texto, nome  # a declaração aparece em todos os estados
    if nome != "inicial":
        assert "Ela não decide se houve fraude." in texto and "posição mais alta que" in texto, nome
        assert len(ajuda) == 4, (nome, len(ajuda))  # 4 textos de ajuda, todos com 2 frases
        for a in ajuda:
            assert len(frases(a)) == 2, a
        assert "Modo demonstração" in texto if nome == "demonstração" else "Modo demonstração" not in texto

print("ok (estados verificados: " + ", ".join(estados) + ")")
