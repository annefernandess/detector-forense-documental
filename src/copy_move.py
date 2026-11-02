"""
Copy-move (cópia dentro do mesmo documento) por correlação normalizada, o filtro
casado visto em aula: a região suspeita é usada como máscara e comparada com cada
posição da imagem. Se ela foi copiada de outro ponto do documento, a correlação com
esse ponto fica próxima de 1.
Fundamentação teórica (correlação de Pearson, filtro casado, limitações):
report/fundamentacao_copy_move.md (slides 260-272 do material da disciplina).
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from dataset_loader import load_gray
from evaluation import generate_pseudo_region, get_text_mask


def _clip_bbox(bbox, shape) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x, y = max(0, x), max(0, y)
    return x, y, min(w, shape[1] - x), min(h, shape[0] - y)


def _overlaps(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def correlation_map(gray: np.ndarray, bbox) -> tuple[np.ndarray | None, tuple | None]:
    """
    Correlação normalizada (coeficiente de Pearson em cada posição) entre a região
    `bbox` e a imagem. Posições que sobrepõem a própria região e janelas lisas
    (correlação indefinida) valem -1. Retorna (mapa, bbox recortada) ou (None, None)
    se a região for lisa.
    """
    x, y, w, h = _clip_bbox(bbox, gray.shape)
    template = gray[y:y + h, x:x + w]
    if template.size == 0 or template.std() == 0:
        return None, None
    corr = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    corr = np.nan_to_num(corr, nan=-1.0, posinf=-1.0, neginf=-1.0)
    corr[max(0, y - h + 1):y + h, max(0, x - w + 1):x + w] = -1.0
    return corr, (x, y, w, h)


def _peak(corr: np.ndarray, box) -> tuple[float, tuple[int, int, int, int]]:
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    return float(corr[py, px]), (int(px), int(py), box[2], box[3])


def correlation_peak(gray: np.ndarray, bbox) -> tuple[float | None, tuple | None]:
    """
    Filtro casado: maior correlação da região com outro ponto da imagem.
    Retorna (pico, bbox do ponto de maior correlação) ou (None, None) se a
    região for lisa.
    """
    corr, box = correlation_map(gray, bbox)
    return (None, None) if corr is None else _peak(corr, box)


def copy_move_region_score(image_path, regions: list) -> tuple[float | None, float | None]:
    """
    Score no formato de `evaluate_region_technique`: maior pico de correlação entre
    as regiões modificadas (áreas originais são ignoradas) e o pico de uma região de
    controle com texto, do tamanho da maior região, na mesma imagem.
    """
    gray = load_gray(image_path)
    targets = [r for r in regions if r.is_original_area is not True]
    peaks = [p for p, _ in (correlation_peak(gray, r.bbox) for r in targets) if p is not None]
    text_mask = get_text_mask(image_path)
    if not peaks or text_mask is None:
        return None, None
    _, _, w, h = _clip_bbox(max(targets, key=lambda r: r.bbox[2] * r.bbox[3]).bbox, gray.shape)
    control = generate_pseudo_region(gray.shape[1], gray.shape[0], w, h, regions, text_mask, random)
    if control is None:
        return max(peaks), None
    return max(peaks), correlation_peak(gray, control.bbox)[0]


def locate_peaks(sample) -> list[tuple]:
    """
    Para cada região modificada, indica onde cai o pico de correlação: "origem"
    (área original anotada), "outra região modificada", "fora das anotações" ou
    "sem origem anotada". Retorna (região, pico, destino) por região.
    """
    gray = load_gray(sample.image_path)
    origins = [r.bbox for r in sample.regions if r.is_original_area is True]
    located = []
    for r in sample.regions:
        if r.is_original_area is True:
            continue
        peak, loc = correlation_peak(gray, r.bbox)
        if peak is None:
            continue
        others = [o.bbox for o in sample.regions if o is not r and o.is_original_area is not True]
        if not origins:
            dest = "sem origem anotada"
        elif any(_overlaps(loc, o) for o in origins):
            dest = "origem"
        elif any(_overlaps(loc, o) for o in others):
            dest = "outra região modificada"
        else:
            dest = "fora das anotações"
        located.append((r, peak, dest))
    return located


def visualize_copy_move(image_path, bbox, title: str | None = None, axes=None):
    """
    Plota, em 1 linha x 3 colunas:
        [0] imagem com a região (vermelho) e o ponto de maior correlação (verde)
        [1] região ampliada
        [2] trecho de maior correlação ampliado
    Se `axes` for None, cria uma figura nova. Retorna a figura.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1, 1]})
    else:
        fig = axes[0].figure

    gray = load_gray(image_path)
    corr, box = correlation_map(gray, bbox)
    peak, loc = _peak(corr, box)

    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    for (x, y, w, h), color in ((box, "red"), (loc, "lime")):
        axes[0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2))
    axes[0].set_title(title or "Região (vermelho) e maior correlação (verde)", fontsize=9)

    for ax, (x, y, w, h), name in ((axes[1], box, "região"),
                                   (axes[2], loc, f"maior correlação (r = {peak:.3f})")):
        ax.imshow(gray[y:y + h, x:x + w], cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=9)

    for ax in axes:
        ax.axis("off")
    return fig



def _text_components(gray_u8: np.ndarray, min_side: int, max_side: int, min_area: int) -> list:
    """
    Caixas (x, y, w, h) dos componentes conexos de texto, na máscara de Otsu. Um
    componente é um caractere ou um pedaço de palavra; comparar uma caixa de
    componente com o restante da imagem evita o problema do alinhamento de grade:
    o contorno de um caractere copiado coincide com o do original.
    """
    _, mask = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(mask) > mask.size / 2:
        mask = cv2.bitwise_not(mask)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for x, y, w, h, area in stats[1:]:
        if min_side <= w <= max_side and min_side <= h <= max_side and area >= min_area:
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def _near(a, b, tol: int) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol

def _dedupe_glyph_pairs(pairs: list, tol: int) -> list:
    """
    Remove pares redundantes. Um par é redundante se origem e destino estão a até
    `tol` pixels de um par já mantido (na ordem direta ou inversa). Mantém o de maior
    correlação. Compara só com os pares de células vizinhas (de lado `tol`), e não com
    todos os mantidos.
    """
    index = {}

    def keys(point):
        bx, by = point[0] // tol, point[1] // tol
        return [(bx + i, by + j) for i in (-1, 0, 1) for j in (-1, 0, 1)]

    kept = []
    for src, dst, r in sorted(pairs, key=lambda p: -p[2]):
        candidates = {id(e): e for k in keys(src) + keys(dst) for e in index.get(k, [])}
        redundant = any((_near(src, s, tol) and _near(dst, d, tol))
                        or (_near(src, d, tol) and _near(dst, s, tol))
                        for s, d, _ in candidates.values())
        if not redundant:
            entry = (src, dst, r)
            kept.append(entry)
            for k in keys(src) + keys(dst):
                index.setdefault(k, []).append(entry)
    return kept


def _merge_boxes(boxes: list) -> list:
    """Une retângulos que se sobrepõem até não restar sobreposição."""
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        out = []
        for b in merged:
            for i, m in enumerate(out):
                if _overlaps(b, m):
                    x, y = min(b[0], m[0]), min(b[1], m[1])
                    X = max(b[0] + b[2], m[0] + m[2])
                    Y = max(b[1] + b[3], m[1] + m[3])
                    out[i] = (x, y, X - x, Y - y)
                    changed = True
                    break
            else:
                out.append(b)
        merged = out
    return merged

def blind_copy_move(gray: np.ndarray, threshold: float, min_side: int = 6, max_side: int = 60,
                    min_area: int = 30, width: int = 1000, min_dist: int = 8) -> tuple[list, list]:
    """
    Busca cega de copy-move, sem anotações.

    Escala: a imagem é redimensionada para largura `width` (proporção preservada) antes
    da busca, para que caracteres de digitalizações de resoluções diferentes tenham
    tamanho comparável. Imagens maiores são reduzidas com INTER_AREA; menores são
    ampliadas com INTER_LINEAR (INTER_AREA amplia como vizinho mais próximo).
    Altura mediana dos componentes de texto em 1000 px, no val: 27 px nas autênticas e
    26 px nas adulteradas (ver notebook 03, seção 7).

    Patches: caixas dos componentes conexos de texto com lado entre `min_side` e
    `max_side` e área de pelo menos `min_area` (na escala de `width`). Esses limites
    descartam pontos e traços (menores) e linhas inteiras ou blocos (maiores).
    Compara-se cada par de componentes de mesmo tamanho (w, h) pela correlação de
    Pearson de seus pixels, que é a correlação normalizada (TM_CCOEFF_NORMED) entre
    duas janelas de mesmo tamanho. Pares com centros a menos de `min_dist` pixels são
    descartados, para que caracteres vizinhos do mesmo traço não contem como cópia.
    Limitação: uma cópia fundida com caracteres vizinhos não tem caixa de mesmo
    tamanho e não é encontrada. Comparar com a imagem inteira (matchTemplate) daria
    o mesmo valor, mas levava cerca de 80 s por imagem em 1000 px.

    Retorna (pares, caixas), em coordenadas da imagem original:
        pares: lista de ((x, y, w, h), (u, v, w, h), correlação) com correlação >=
               `threshold`, sem redundância;
        caixas: retângulos candidatos (origem e destino), unidos quando se sobrepõem.
    """
    scale = width / gray.shape[1]
    small = cv2.resize(gray, (width, round(gray.shape[0] * scale)),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    small = small.astype(np.float32)
    templates = _text_components(small.astype(np.uint8), min_side, max_side, min_area)

    groups = {}
    for x, y, w, h in templates:
        patch = small[y:y + h, x:x + w]
        if patch.std() > 0:
            groups.setdefault((w, h), []).append((x, y, patch))

    pairs = []
    for (w, h), items in groups.items():
        if len(items) < 2:
            continue
        X = np.stack([p.ravel() for _, _, p in items]).astype(np.float64)
        X -= X.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(X, axis=1)
        valid = norms > 1e-9
        if valid.sum() < 2:
            continue
        Z = np.where(valid[:, None], X / np.where(valid, norms, 1.0)[:, None], 0.0)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            C = Z @ Z.T
        C = np.nan_to_num(C, nan=-1.0, posinf=-1.0, neginf=-1.0)
        C[~valid, :] = -1.0
        C[:, ~valid] = -1.0
        for i, j in zip(*np.where(np.triu(C >= threshold, k=1))):
            (xi, yi, _), (xj, yj, _) = items[i], items[j]
            if max(abs(xi - xj), abs(yi - yj)) < min_dist:
                continue
            pairs.append(((xi, yi, w, h), (xj, yj, w, h), float(C[i, j])))

    kept = _dedupe_glyph_pairs(pairs, tol=4)
    boxes = _merge_boxes([box for src, dst, _ in kept for box in (src, dst)])

    back = lambda b: tuple(round(v / scale) for v in b)
    kept = [(back(src), back(dst), r) for src, dst, r in kept]
    return kept, [back(b) for b in boxes]


def blind_copy_move_for_path(image_path, threshold: float, **kwargs) -> tuple[list, list]:
    """`blind_copy_move` a partir do caminho da imagem. Módulo-nível para uso com multiprocessing."""
    return blind_copy_move(load_gray(image_path), threshold, **kwargs)

def coverage(region_bbox, box) -> float:
    """Fração da área da região anotada coberta pela caixa candidata (0 a 1)."""
    rx, ry, rw, rh = region_bbox
    bx, by, bw, bh = box
    iw = max(0, min(rx + rw, bx + bw) - max(rx, bx))
    ih = max(0, min(ry + rh, by + bh) - max(ry, by))
    return (iw * ih) / (rw * rh) if rw * rh > 0 else 0.0


def filter_pairs(pairs: list, threshold: float, tol: int = 4) -> list:
    """Reaplica um limiar mais alto a pares já calculados e remove redundância."""
    return _dedupe_glyph_pairs([p for p in pairs if p[2] >= threshold], tol)


def candidate_boxes(pairs: list) -> list:
    """Caixas candidatas (origem e destino) de uma lista de pares."""
    return [box for src, dst, _ in pairs for box in (src, dst)]


def region_covered(region_bbox, boxes: list, min_cover: float = 0.5) -> bool:
    """True se alguma caixa candidata cobre pelo menos `min_cover` da região anotada."""
    return any(coverage(region_bbox, box) >= min_cover for box in boxes)
