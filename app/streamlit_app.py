"""
Interface do detector: triagem de recibos digitalizados, em três etapas (enviar, ver o resultado,
entender o porquê). Carrega o classificador candidato congelado e o limiar de results/; não treina,
não recalibra e não lê o test. Rodar a partir da raiz do projeto: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import random
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from PIL import Image

import textos as T
from dataset_loader import load_dataset
from estilo import CSS, etapas_html, figura_html, resultado_html
from explicacao import resumo
from overlays import desenhar_gabarito, painel_copy_move, painel_ela, painel_ghost, painel_original
from pipeline import analisar_imagem, metricas_candidato

st.set_page_config(page_title=T.TITULO, page_icon="🔎", layout="centered")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

EXTENSOES = (".png", ".jpg", ".jpeg")


@st.cache_data(show_spinner=False)
def analisar_bytes(conteudo: bytes, sufixo: str) -> dict:
    """Analisa uma imagem enviada pelo usuário (grava um arquivo temporário)."""
    with tempfile.NamedTemporaryFile(suffix=sufixo, delete=False) as arquivo:
        arquivo.write(conteudo)
    try:
        return analisar_imagem(arquivo.name)
    finally:
        Path(arquivo.name).unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def analisar_caminho(caminho: str) -> dict:
    """Analisa uma imagem do conjunto de dados (exemplos)."""
    return analisar_imagem(caminho)


@st.cache_data(show_spinner=False)
def exemplos_val() -> dict:
    """Exemplos do val: 3 de cada tipo (seed 42), embaralhados. {nome do arquivo: amostra}. Vazio sem o dataset."""
    if not (ROOT / "data/findit2/val").exists():
        return {}
    amostras = load_dataset(str(ROOT / "data/findit2"), "val")
    rng = random.Random(42)
    escolhidas = (rng.sample([s for s in amostras if not s.is_tampered], 3)
                  + rng.sample([s for s in amostras if s.is_tampered and s.regions], 3))
    rng.shuffle(escolhidas)
    return {s.image_path.name: s for s in escolhidas}


def validar(arquivo) -> str | None:
    """Mensagem de erro em português se o arquivo não for PNG ou JPG legível, ou for grande demais."""
    if arquivo.size > T.LIMITE_MB * 1024 * 1024:
        return T.ERRO_TAMANHO
    if Path(arquivo.name).suffix.lower() not in EXTENSOES:
        return T.ERRO_FORMATO
    try:
        with Image.open(arquivo) as imagem:
            formato = imagem.format
            imagem.verify()
    except Exception:
        return T.ERRO_ABRIR
    arquivo.seek(0)
    return None if formato in ("PNG", "JPEG") else T.ERRO_FORMATO


def ajuda(texto: str):
    with st.expander(T.AJUDA_ROTULO):
        st.write(texto)


demo = st.query_params.get("demo") == "1"

st.title(T.TITULO)
st.markdown(f'<p class="subtitulo">{T.SUBTITULO}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitulo"><strong>{T.DECLARACAO}</strong></p>', unsafe_allow_html=True)
if demo:
    st.markdown(f'<div class="demo-aviso" role="note">⚠ {T.DEMO_AVISO}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="link-discreto"><a href="?" target="_self">{T.DEMO_SAIR}</a></p>', unsafe_allow_html=True)
etapas = st.empty()
etapas.markdown(etapas_html(1), unsafe_allow_html=True)

# Etapa 1: enviar a imagem ou escolher um exemplo
st.header("1. " + T.ETAPAS[0][0])
st.write(T.UPLOAD_ORIENTACAO)
arquivo = st.file_uploader("Escolher uma imagem", type=None, label_visibility="collapsed")
exemplos = exemplos_val()
escolhido = st.session_state.get("exemplo") or st.query_params.get("exemplo")
if exemplos:
    st.markdown(f"**{T.EXEMPLOS_TITULO}**")
    st.caption(T.EXEMPLOS_AVISO)
    colunas = st.columns(3)
    for i, nome in enumerate(exemplos):
        if colunas[i % 3].button(f"Exemplo {i + 1}", key=f"exemplo_{i}", width="stretch"):
            st.session_state["exemplo"] = escolhido = nome
if not demo:
    st.markdown(f'<p class="link-discreto"><a href="?demo=1" target="_self">{T.DEMO_LINK}</a></p>', unsafe_allow_html=True)

if arquivo is not None:
    erro = validar(arquivo)
    if erro:
        st.error(erro)
        st.stop()
    fonte, amostra = "upload", None
elif escolhido in exemplos:
    fonte, amostra = "exemplo", exemplos[escolhido]
    st.write(f"Exemplo selecionado: **Exemplo {list(exemplos).index(escolhido) + 1}**.")
else:
    st.info(T.AGUARDANDO)
    st.stop()

try:
    with st.spinner(T.ANALISANDO):
        if fonte == "upload":
            r = analisar_bytes(arquivo.getvalue(), Path(arquivo.name).suffix.lower())
        else:
            r = analisar_caminho(str(amostra.image_path))
except Exception:
    traceback.print_exc()
    st.error(T.ERRO_ANALISE)
    st.stop()

etapas.markdown(etapas_html(2), unsafe_allow_html=True)
cinza, escala = r["cinza_exibicao"], r["escala"]

# Imagem analisada (o gabarito só aparece no modo demonstração, em exemplos do conjunto de dados)
mostrar_gabarito = demo and fonte == "exemplo"
previa = painel_original(cinza)
if mostrar_gabarito:
    previa = desenhar_gabarito(previa, amostra.regions, escala)
st.subheader(T.PREVIA_TITULO)
st.markdown(figura_html(previa, T.PREVIA_ALT_DEMO if mostrar_gabarito else T.PREVIA_ALT, classe="previa"), unsafe_allow_html=True)
if mostrar_gabarito:
    st.caption(T.DEMO_LEGENDA if amostra.regions else T.DEMO_SEM_GABARITO)

# Etapa 2: resultado
st.header("2. " + T.ETAPAS[1][0])
faixa, icone = T.FAIXAS[r["faixa"]]
st.markdown(resultado_html(faixa, icone, resumo(r["faixa"], len(r["copy_move_pares"]))), unsafe_allow_html=True)

st.subheader(T.COMO_LER_TITULO)
st.markdown(T.COMO_LER_FAIXA)
st.markdown(f"**{T.COMO_LER_REGRA}**")
if r["posicao"] is not None:
    st.markdown(f'<div class="posicao" role="status">{T.posicao_html(r["posicao"])}</div>', unsafe_allow_html=True)
else:
    st.write(T.POSICAO_ALT)
ajuda(T.POSICAO_AJUDA)
st.write(T.numeros_medidos(metricas_candidato()))

# Etapa 3: mapas
st.header("3. " + T.ETAPAS[2][0])
st.write(T.MAPAS_INTRO)
colunas = st.columns(3)
for coluna, chave in zip(colunas, ("ela", "ghost", "copia")):
    texto = T.MAPAS[chave]
    with coluna:
        st.subheader(texto["titulo"])
        if chave == "ghost" and not r["ghost_historico"]:
            st.markdown(figura_html(painel_original(cinza), texto["alt_vazio"]), unsafe_allow_html=True)
            st.markdown(f'<div class="mapa-vazio">{texto["vazio"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="legenda">{texto["legenda"]}</p>', unsafe_allow_html=True)
            espaco = st.empty()
            transparencia = st.slider(T.TRANSPARENCIA, 0, 100, 50, key=f"transp_{chave}", help=T.TRANSPARENCIA_AJUDA)
            opacidade = 1 - transparencia / 100
            if chave == "ela":
                imagem, alt = painel_ela(cinza, r["mapa_ela"], opacidade), texto["alt"]
            elif chave == "ghost":
                imagem = painel_ghost(cinza, r["ghost_mapa"], escala, opacidade, r["ghost_qualidades"])
                alt = texto["alt"]
            else:
                pares = r["copy_move_pares"]
                imagem = painel_copy_move(cinza, pares, escala, opacidade)
                alt = texto["alt"] if pares else texto["alt_vazio"]
            espaco.markdown(figura_html(imagem, alt), unsafe_allow_html=True)
            if chave == "copia" and not r["copy_move_pares"]:
                st.markdown(f'<div class="mapa-vazio">{texto["vazio"]}</div>', unsafe_allow_html=True)
        st.write(texto["mostra"])
        st.write(texto["nao_mostra"])
        ajuda(texto["ajuda"])

# Limitações
st.header(T.NAO_FAZ_TITULO)
for item in T.NAO_FAZ:
    st.markdown(f"- {item}")
st.caption(T.RODAPE)
