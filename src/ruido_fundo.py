"""
Mapas de ruído local (técnica A) e de fundo do papel (técnica B), em blocos de lado B, e a avaliação
deles como mapa de região. Nenhuma anotação entra nos mapas nem nas features; as anotações servem só
para medir a localização.

Fundo: pixels fora da máscara de Otsu do texto, dilatada (dilatação é método de apoio; morfologia
matemática não aparece nos slides da disciplina). Só blocos inteiros contam; as sobras da borda são
ignoradas. Um bloco com menos de LIMIAR_FUNDO de fundo é ignorado.

A: resíduo = imagem - mediana 3 x 3 (copy_move_residuo.residuo); por bloco, sigma = MAD x 1,4826 dos
   pixels de fundo; mapa = z robusto de log(sigma) (blocos com sigma = 0 não têm log e são ignorados).
B: polinômio de grau 2 em x e y (coordenadas em [-1, 1]) ajustado por mínimos quadrados a pixels de
   fundo (no máximo MAX_PIXELS_AJUSTE, sorteados com seed 42); resíduo = imagem - superfície; por bloco,
   média do |resíduo| nos pixels de fundo; mapa = z robusto dessa média.
z robusto = (valor - mediana) / (1,4826 x MAD), sobre os blocos válidos da imagem. Sem MIN_BLOCOS
blocos válidos, ou com MAD = 0, o mapa fica indefinido (None).

Grade e critérios congelados em results/grade_ruido_fundo.json (hash deste arquivo).
"""

from __future__ import annotations

import functools
import random
from multiprocessing import get_context

import cv2
import numpy as np
from PIL import Image

from copy_move_residuo import residuo
from dataset_loader import load_gray
from evaluation import evaluate_region_technique, generate_pseudo_region, get_text_mask
from gerar_copy_move_sintetico import mascara_texto

TAMANHOS = (32, 64)
KERNEL_DILATACAO = 5
LIMIAR_FUNDO = 0.5
MIN_BLOCOS = 10
LIMIAR_Z = 3.0
MAX_PIXELS_AJUSTE = 200_000
MAD_ESCALA = 1.4826
SEED = 42
N_PSEUDO = 20
NOMES = ("local", "fundo")


def mascara_fundo(gray: np.ndarray) -> np.ndarray:
    """Pixels de fundo: fora da máscara de Otsu do texto dilatada (kernel KERNEL_DILATACAO)."""
    texto = mascara_texto(gray).astype(np.uint8)
    return cv2.dilate(texto, np.ones((KERNEL_DILATACAO, KERNEL_DILATACAO), np.uint8)) == 0


def _blocos(a: np.ndarray, b: int) -> np.ndarray:
    """(ny, nx, b*b): os blocos inteiros de lado b, em linhas."""
    ny, nx = a.shape[0] // b, a.shape[1] // b
    return a[:ny * b, :nx * b].reshape(ny, b, nx, b).transpose(0, 2, 1, 3).reshape(ny, nx, b * b)


def z_robusto(valores: np.ndarray) -> np.ndarray | None:
    """z robusto dos valores; None com menos de MIN_BLOCOS valores ou MAD = 0."""
    if len(valores) < MIN_BLOCOS:
        return None
    mediana = np.median(valores)
    mad = MAD_ESCALA * np.median(np.abs(valores - mediana))
    return None if mad == 0 else (valores - mediana) / mad


def _mapa_a_partir_de(valores_por_bloco: np.ndarray, valido: np.ndarray) -> np.ndarray | None:
    """Mapa (ny, nx) de z robusto, NaN nos blocos inválidos; None se o z for indefinido."""
    z = z_robusto(valores_por_bloco)
    if z is None:
        return None
    mapa = np.full(valido.shape, np.nan, np.float32)
    mapa[valido] = z
    return mapa


def mapa_ruido_local(res: np.ndarray, fundo: np.ndarray, b: int) -> np.ndarray | None:
    """Técnica A: z robusto de log(sigma) do resíduo, por bloco de fundo."""
    rb, fb = _blocos(res, b), _blocos(fundo, b)
    valido = fb.mean(axis=2) >= LIMIAR_FUNDO
    if not valido.any():
        return None
    valores = np.where(fb[valido], rb[valido], np.nan)
    mediana = np.nanmedian(valores, axis=1)
    sigma = MAD_ESCALA * np.nanmedian(np.abs(valores - mediana[:, None]), axis=1)
    com_log = sigma > 0
    final = np.zeros_like(valido)
    final[valido] = com_log
    return _mapa_a_partir_de(np.log(sigma[com_log]), final)


def superficie_polinomial(gray: np.ndarray, fundo: np.ndarray) -> np.ndarray | None:
    """Polinômio de grau 2 ajustado por mínimos quadrados aos pixels de fundo, avaliado na imagem toda.
    None com menos de 6 pixels de fundo."""
    altura, largura = gray.shape
    ys, xs = np.nonzero(fundo)
    if len(ys) < 6:
        return None
    if len(ys) > MAX_PIXELS_AJUSTE:
        escolha = np.random.default_rng(SEED).choice(len(ys), MAX_PIXELS_AJUSTE, replace=False)
        ys, xs = ys[escolha], xs[escolha]
    x, y = xs / max(largura - 1, 1) * 2 - 1, ys / max(altura - 1, 1) * 2 - 1
    termos = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
    c = np.linalg.lstsq(termos, gray[ys, xs].astype(np.float64), rcond=None)[0].astype(np.float32)
    xg = np.linspace(-1, 1, largura, dtype=np.float32)[None, :]
    yg = np.linspace(-1, 1, altura, dtype=np.float32)[:, None]
    return c[0] + c[1] * xg + c[2] * yg + c[3] * xg * xg + c[4] * yg * xg + c[5] * yg * yg


def mapa_fundo_papel(gray: np.ndarray, fundo: np.ndarray, b: int) -> np.ndarray | None:
    """Técnica B: z robusto da média do |resíduo da superfície| por bloco de fundo."""
    superficie = superficie_polinomial(gray, fundo)
    if superficie is None:
        return None
    absoluto = np.abs(gray.astype(np.float32) - superficie)
    ab, fb = _blocos(absoluto, b), _blocos(fundo, b)
    valido = fb.mean(axis=2) >= LIMIAR_FUNDO
    if not valido.any():
        return None
    medias = (ab[valido] * fb[valido]).sum(axis=1) / fb[valido].sum(axis=1)
    return _mapa_a_partir_de(medias, valido)


def mapas_imagem(gray: np.ndarray, tamanhos=TAMANHOS) -> dict:
    """Mapas de uma imagem: {"local_32": ..., "local_64": ..., "fundo_32": ..., "fundo_64": ...} (None se indefinido)."""
    fundo = mascara_fundo(gray)
    res = residuo(gray, 3)
    saida = {}
    for b in tamanhos:
        saida[f"local_{b}"] = mapa_ruido_local(res, fundo, b)
        saida[f"fundo_{b}"] = mapa_fundo_papel(gray, fundo, b)
    return saida


def features_mapa(mapa: np.ndarray | None) -> tuple:
    """(percentil 99 de |z|, fração de blocos com |z| > LIMIAR_Z) nos blocos válidos; (None, None) se indefinido."""
    if mapa is None:
        return None, None
    z = np.abs(mapa[~np.isnan(mapa)])
    return float(np.percentile(z, 99)), float((z > LIMIAR_Z).mean())


def score_regiao(mapa: np.ndarray | None, caixa, b: int) -> float | None:
    """Maior |z| entre os blocos válidos que tocam a caixa (x, y, w, h); None se não houver."""
    if mapa is None:
        return None
    x, y, w, h = caixa
    ny, nx = mapa.shape
    x0, y0 = max(x, 0) // b, max(y, 0) // b
    x1, y1 = min(x + w, nx * b) - 1, min(y + h, ny * b) - 1
    if x1 < 0 or y1 < 0 or x0 > nx - 1 or y0 > ny - 1:
        return None
    sub = mapa[y0:min(y1 // b, ny - 1) + 1, x0:min(x1 // b, nx - 1) + 1]
    valores = np.abs(sub[~np.isnan(sub)])
    return float(valores.max()) if len(valores) else None


def _mapas_caminho(caminho):
    cv2.setNumThreads(1)
    return str(caminho), mapas_imagem(load_gray(caminho))


def mapas_lote(caminhos: list, processos: int = 8) -> dict:
    """Mapas de cada imagem (caminho -> dict de mapas), em paralelo."""
    with get_context("fork").Pool(processos) as pool:
        return dict(pool.map(_mapas_caminho, caminhos))


def nomes_features(tamanhos=TAMANHOS) -> list:
    """Nomes das 8 colunas de features, na ordem técnica, B, (p99, fração)."""
    return [f"{n}_{b}_{s}" for n in NOMES for b in tamanhos for s in ("p99", "frac")]


def features_lote(mapas_por_imagem: dict, tamanhos=TAMANHOS) -> dict:
    """caminho -> {coluna: valor ou None} com p99 e fração de cada mapa."""
    saida = {}
    for caminho, mapas in mapas_por_imagem.items():
        linha = {}
        for n in NOMES:
            for b in tamanhos:
                linha[f"{n}_{b}_p99"], linha[f"{n}_{b}_frac"] = features_mapa(mapas[f"{n}_{b}"])
        saida[caminho] = linha
    return saida


def _alvos(amostra) -> list:
    """Regiões modificadas da amostra (as marcadas como origem ficam de fora; sem nenhuma, todas, como em evaluation.py)."""
    return [r for r in amostra.regions if r.is_original_area is not True] or amostra.regions


def auc_regioes(amostras: list, mapas_por_imagem: dict, nome: str, b: int, n_boot: int = 2000) -> dict:
    """
    AUC no nível de região contra o acaso. Por imagem adulterada: a região real vale o maior score entre
    as regiões modificadas; as pseudo-regiões vêm de generate_pseudo_region (mesmo tamanho da maior região
    modificada, com texto, sem sobreposição com as anotadas), N_PSEUDO sorteios com seed 42, como em
    evaluate_region_technique. A AUC da imagem é a fração de pseudo-regiões com score menor que o da região
    real (empate vale 1/2); a AUC final é a média entre imagens, com IC de bootstrap sobre as imagens.
    Imagens com região real indefinida ou com menos de 10 pseudo-regiões válidas ficam de fora (contadas).
    """
    rng = random.Random(SEED)
    chave, aucs = f"{nome}_{b}", []
    exclusoes = {"mapa_indefinido": 0, "regiao_real_sem_bloco": 0, "poucas_pseudo_validas": 0}
    for s in amostras:
        if not (s.is_tampered and s.regions):
            continue
        mapa = mapas_por_imagem[str(s.image_path)][chave]
        texto = get_text_mask(str(s.image_path))
        alvos = _alvos(s)
        reais = [v for v in (score_regiao(mapa, r.bbox, b) for r in alvos) if v is not None]
        if mapa is None:
            exclusoes["mapa_indefinido"] += 1
            continue
        if not reais:
            exclusoes["regiao_real_sem_bloco"] += 1
            continue
        ref = max(alvos, key=lambda r: r.bbox[2] * r.bbox[3])
        with Image.open(s.image_path) as img:
            largura, altura = img.size
        pseudo = []
        for _ in range(N_PSEUDO):
            p = generate_pseudo_region(largura, altura, ref.bbox[2], ref.bbox[3], s.regions, texto, rng)
            valor = None if p is None else score_regiao(mapa, p.bbox, b)
            if valor is not None:
                pseudo.append(valor)
        if len(pseudo) < 10:
            exclusoes["poucas_pseudo_validas"] += 1
            continue
        real, pseudo = max(reais), np.array(pseudo)
        aucs.append(float((pseudo < real).mean() + 0.5 * (pseudo == real).mean()))
    aucs = np.array(aucs)
    boot = np.random.default_rng(SEED)
    medias = [aucs[boot.integers(0, len(aucs), len(aucs))].mean() for _ in range(n_boot)]
    return {"tecnica": nome, "B": b, "n_imagens": int(len(aucs)), "auc": float(aucs.mean()),
            "ic_min": float(np.percentile(medias, 2.5)), "ic_max": float(np.percentile(medias, 97.5)),
            "exclusoes": exclusoes}


def hit_rate_evaluation(amostras: list, mapas_por_imagem: dict, nome: str, b: int) -> dict:
    """A linha de base de acaso existente (evaluation.evaluate_region_technique): acerto real (região > controle
    com texto) contra o acaso de pseudo-regiões, com o score de região deste módulo."""
    chave = f"{nome}_{b}"

    @functools.lru_cache(maxsize=2)
    def texto(caminho):
        return get_text_mask(caminho)

    def score(caminho, regioes):
        mapa = mapas_por_imagem[str(caminho)][chave]
        alvos = [r for r in regioes if r.is_original_area is not True]
        reais = [v for v in (score_regiao(mapa, r.bbox, b) for r in alvos) if v is not None]
        if not reais:
            return None, None
        x, y, w, h = max(alvos, key=lambda r: r.bbox[2] * r.bbox[3]).bbox
        with Image.open(caminho) as img:
            largura, altura = img.size
        controle = generate_pseudo_region(largura, altura, w, h, regioes, texto(str(caminho)), random)
        return (max(reais), None if controle is None else score_regiao(mapa, controle.bbox, b))

    return evaluate_region_technique(amostras, score, lambda s_in, s_out: s_in > s_out)
