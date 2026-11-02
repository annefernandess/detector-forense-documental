"""Estilo e peças HTML da interface: paleta em tokens :root, etapas, faixa de resultado e figuras com texto alternativo."""

from __future__ import annotations

import base64
import html
import io

import cv2
import numpy as np
from PIL import Image

from textos import ETAPAS

CSS = """
:root {
  --fundo: #FAFAF7;
  --superficie: #FFFFFF;
  --texto: #1F2933;
  --texto-suave: #46515C;
  --borda: #B8C2CC;
  --acao: #1D4ED8;
  --revisar-fundo: #FFF1DB;
  --revisar-borda: #B45309;
  --revisar-texto: #5C2A03;
  --neutro-fundo: #E8EDF2;
  --neutro-borda: #4B6075;
  --neutro-texto: #1B2733;
  --aviso-fundo: #FFF7CC;
  --aviso-texto: #4A3B00;
  --raio: 12px;
}
.stApp { background: var(--fundo); color: var(--texto); }
.block-container, [data-testid="stMainBlockContainer"] {
  max-width: 900px !important; padding: 2.5rem 1.25rem 4rem !important; margin: 0 auto;
}
@media (max-width: 640px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem 3rem !important; }
}
h1 { font-size: 2.1rem; line-height: 1.2; margin-bottom: .25rem; }
h2 { font-size: 1.5rem; margin-top: 2.5rem; }
h3 { font-size: 1.15rem; }
.stApp p, .stApp li { font-size: 1.05rem; line-height: 1.65; }
.subtitulo { font-size: 1.15rem; color: var(--texto-suave); margin: 0 0 1rem; }
a { color: var(--acao); }
.link-discreto { font-size: .9rem; color: var(--texto-suave); }
.link-discreto a { color: var(--texto-suave); }

.stApp ol.etapas { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: .75rem;
          list-style: none; padding: 0 !important; padding-inline-start: 0 !important; margin: 1.25rem 0 2rem !important; }
.stApp ol.etapas > li.etapa {
  margin: 0 !important; padding: 1.1rem 1.1rem 1.1rem 1.1rem !important; align-items: center !important;
  overflow: visible; box-sizing: border-box;
}
.stApp ol.etapas > li.etapa .num { margin: 0 !important; flex: none; }
.etapa { display: flex; gap: .75rem; align-items: flex-start; padding: .9rem 1rem; background: var(--superficie);
         border: 2px solid var(--borda); border-radius: var(--raio); }
.etapa--atual { border-color: var(--acao); }
.etapa .num { flex: none; width: 2rem; height: 2rem; display: grid; place-items: center; border-radius: 50%;
              background: var(--texto); color: #FFFFFF; font-weight: 700; }
.etapa--atual .num { background: var(--acao); }
.etapa strong { display: block; color: var(--texto); }
.etapa .desc { display: block; color: var(--texto-suave); font-size: .95rem; line-height: 1.4; }
.so-leitor { position: absolute; left: -9999px; }

.resultado { display: flex; gap: 1rem; align-items: flex-start; padding: 1.25rem 1.5rem; border: 3px solid;
             border-radius: var(--raio); margin: .5rem 0 1.5rem; }
.resultado--revisar { background: var(--revisar-fundo); border-color: var(--revisar-borda); color: var(--revisar-texto); }
.resultado--neutro { background: var(--neutro-fundo); border-color: var(--neutro-borda); color: var(--neutro-texto); }
.resultado .icone { font-size: 2rem; line-height: 1.2; }
.resultado .faixa { font-size: 1.6rem; font-weight: 700; margin: 0 0 .25rem; color: inherit; }
.resultado .resumo { margin: 0; color: inherit; }

.demo-aviso { background: var(--aviso-fundo); color: var(--aviso-texto); border: 2px solid var(--aviso-texto);
              border-radius: var(--raio); padding: .75rem 1rem; margin: 0 0 1rem; font-weight: 600; }
figure.mapa { margin: .5rem 0; }
figure.mapa img { display: block; width: 100%; height: auto; border: 1px solid var(--borda); border-radius: 8px; }
.mapa-vazio { background: var(--neutro-fundo); color: var(--neutro-texto); border: 2px dashed var(--neutro-borda);
              border-radius: 8px; padding: .75rem; margin: .5rem 0; }
.legenda { color: var(--texto-suave); font-size: .95rem; line-height: 1.45; margin: 0 0 .5rem; }

[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stSliderTickBar"] span { color: var(--texto-suave) !important; }
figure.previa { max-width: 440px; }
@media (min-width: 641px) {
  [data-testid="stColumn"] h3 { min-height: 3.4em; }
  .legenda { min-height: 5.4em; }
}

.posicao {
  background: var(--superficie); border: 3px solid var(--acao); border-radius: var(--raio);
  padding: 1rem 1.25rem; margin: .75rem 0 1rem; font-size: 1.3rem; line-height: 1.45; color: var(--texto);
}
.posicao strong { font-size: 1.6rem; color: var(--acao); }

/* Botão de envio em português: o texto original some (font-size 0) e o novo entra por ::after. */
[data-testid="stFileUploaderDropzoneInstructions"] > div > span { font-size: 0; }
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child::after {
  content: "Arraste a imagem para cá"; font-size: 1rem; color: var(--texto);
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:last-child::after {
  content: "PNG ou JPG, até 20 MB"; font-size: .9rem; color: var(--texto-suave);
}
[data-testid="stFileUploaderDropzone"] button { font-size: 0; }
[data-testid="stFileUploaderDropzone"] button::after { content: "Escolher arquivo"; font-size: 1rem; }

a:focus-visible, button:focus-visible, summary:focus-visible, input:focus-visible,
[role="slider"]:focus-visible, [data-testid="stFileUploaderDropzone"]:focus-within {
  outline: 3px solid var(--acao) !important; outline-offset: 2px;
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""


def etapas_html(atual: int) -> str:
    """Lista numerada das três etapas, com a etapa atual destacada (também para leitores de tela)."""
    itens = []
    for i, (titulo, descricao) in enumerate(ETAPAS, 1):
        classe = "etapa etapa--atual" if i == atual else "etapa"
        marca = '<span class="so-leitor"> (etapa atual)</span>' if i == atual else ""
        itens.append(f'<li class="{classe}"><span class="num" aria-hidden="true">{i}</span>'
                     f'<span><strong>{html.escape(titulo)}{marca}</strong>'
                     f'<span class="desc">{html.escape(descricao)}</span></span></li>')
    return f'<ol class="etapas" aria-label="Etapas">{"".join(itens)}</ol>'


def resultado_html(faixa: str, icone: str, resumo: str) -> str:
    """Faixa de resultado: cor, ícone e texto juntos (a cor sozinha não indica nada)."""
    classe = "resultado--revisar" if faixa == "Revisar" else "resultado--neutro"
    return (f'<div class="resultado {classe}" role="status"><span class="icone" aria-hidden="true">{icone}</span>'
            f'<div><p class="faixa">{html.escape(faixa)}</p><p class="resumo">{html.escape(resumo)}</p></div></div>')


def figura_html(rgb: np.ndarray, alt: str, largura: int = 700, classe: str = "mapa") -> str:
    """Imagem embutida (JPEG em base64) com texto alternativo, para ocupar a largura da coluna."""
    if rgb.shape[1] > largura:
        rgb = cv2.resize(rgb, (largura, round(rgb.shape[0] * largura / rgb.shape[1])), interpolation=cv2.INTER_AREA)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, "JPEG", quality=85)
    dados = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (f'<figure class="mapa {classe}"><img src="data:image/jpeg;base64,{dados}" '
            f'alt="{html.escape(alt, quote=True)}"></figure>')
