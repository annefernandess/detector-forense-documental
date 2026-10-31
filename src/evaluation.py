from __future__ import annotations

import random
import statistics
import math
from dataclasses import replace

import cv2
import numpy as np
from PIL import Image

from dataset_loader import DocumentSample, TamperRegion, read_gray_cv2

def wilson_score_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - spread, center + spread

def get_text_mask(image_path: str) -> np.ndarray | None:
    img = read_gray_cv2(image_path)
    if img is None:
        return None
    _, text_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(text_mask) > text_mask.size / 2:
        text_mask = cv2.bitwise_not(text_mask)
    return text_mask

def generate_pseudo_region(
    img_width: int,
    img_height: int,
    ref_w: int,
    ref_h: int,
    avoid_regions: list[TamperRegion],
    text_mask: np.ndarray,
    rng: random.Random,
    max_tries: int = 200,
) -> TamperRegion | None:
    """Gera uma região aleatória do mesmo tamanho que não sobrepõe as evitadas, e contém texto."""
    for _ in range(max_tries):
        x = rng.randint(0, max(0, img_width - ref_w))
        y = rng.randint(0, max(0, img_height - ref_h))

        overlap = False
        for a in avoid_regions:
            ax, ay, aw, ah = a.bbox
            if not (x + ref_w <= ax or x >= ax + aw or y + ref_h <= ay or y >= ay + ah):
                overlap = True
                break
        if overlap:
            continue
            
        c_text = text_mask[y:y+ref_h, x:x+ref_w]
        if np.count_nonzero(c_text) > 0:
            return TamperRegion(
                bbox=(x, y, ref_w, ref_h),
                modification="PSEUDO",
                entity_type=None,
                is_original_area=None,
            )
    return None

def evaluate_region_technique(
    samples: list[DocumentSample],
    score_fn,
    hit_fn,
    chance_trials: int = 20,
    seed: int = 42,
) -> dict:
    """
    Generaliza a avaliação de uma técnica de detecção local em imagens adulteradas.
    A imagem é incluída se a região real for avaliável E pelo menos 10 pseudo-regiões forem válidas.
    """
    rng = random.Random(seed)
    random.seed(seed)

    valid_samples = [s for s in samples if s.is_tampered and s.regions]

    n_valid = 0
    real_hits_total = 0
    
    # Taxa de acaso média de cada imagem, para o cálculo do desvio padrão
    image_chance_rates = []
    
    # Total global, para o intervalo de Wilson
    total_chance_trials_valid = 0
    total_chance_hits = 0
    
    exclusions = {
        "text_mask_failed": 0,
        "real_region_failed": 0,
        "few_valid_pseudo": 0
    }

    for s in valid_samples:
        text_mask = get_text_mask(s.image_path)
        if text_mask is None:
            exclusions["text_mask_failed"] += 1
            continue

        # 1) Avaliação na região real
        s_in, s_out = score_fn(s.image_path, s.regions)
        if s_in is None or s_out is None:
            exclusions["real_region_failed"] += 1
            continue
            
        real_hit = hit_fn(s_in, s_out)

        # 2) Linha de base nula (pseudo-regiões)
        mods = [r for r in s.regions if getattr(r, "is_original_area", None) is not True] or s.regions
        ref = max(mods, key=lambda r: r.bbox[2] * r.bbox[3])

        with Image.open(s.image_path) as img:
            img_width, img_height = img.size

        masked_regions = [replace(r, is_original_area=True) for r in s.regions]

        valid_trial_results = []

        for _ in range(chance_trials):
            pseudo = generate_pseudo_region(
                img_width, img_height, ref.bbox[2], ref.bbox[3], s.regions, text_mask, rng
            )
            if not pseudo:
                continue
                
            ps_in, ps_out = score_fn(s.image_path, [pseudo] + masked_regions)
            if ps_in is None or ps_out is None:
                continue
                
            valid_trial_results.append(hit_fn(ps_in, ps_out))

        if len(valid_trial_results) >= 10:
            n_valid += 1
            if real_hit:
                real_hits_total += 1
            
            hits_in_image = sum(valid_trial_results)
            valid_in_image = len(valid_trial_results)
            
            image_chance_rates.append(hits_in_image / valid_in_image)
            total_chance_hits += hits_in_image
            total_chance_trials_valid += valid_in_image
        else:
            exclusions["few_valid_pseudo"] += 1

    if n_valid == 0:
        return {
            "n_samples": 0,
            "real_hit_rate": 0.0,
            "real_wilson": (0.0, 0.0),
            "chance_hit_rate": 0.0,
            "chance_hit_std": 0.0,
            "chance_wilson": (0.0, 0.0),
            "real_valid": 0,
            "null_valid": 0,
            "real_hits": 0,
            "exclusions": exclusions
        }

    real_rate = real_hits_total / n_valid
    real_wilson = wilson_score_interval(real_rate, n_valid)
    
    chance_mean = statistics.mean(image_chance_rates)
    chance_std = statistics.stdev(image_chance_rates) if n_valid > 1 else 0.0
    chance_global_rate = total_chance_hits / total_chance_trials_valid if total_chance_trials_valid > 0 else 0.0
    chance_wilson = wilson_score_interval(chance_global_rate, total_chance_trials_valid)

    return {
        "n_samples": n_valid,
        "real_hit_rate": real_rate,
        "real_wilson": real_wilson,
        "chance_hit_rate": chance_mean,
        "chance_hit_std": chance_std,
        "chance_wilson": chance_wilson,
        "real_valid": n_valid,
        "null_valid": total_chance_trials_valid,
        "real_hits": real_hits_total,
        "exclusions": exclusions
    }


def youden_threshold(positives, negatives, candidates) -> tuple[float, float, float]:
    """
    Limiar que maximiza o índice de Youden (J = TPR - FPR), decidindo "positivo"
    quando score >= limiar. Em empate, fica o primeiro candidato da lista.
    Retorna (limiar, TPR, FPR).
    """
    pos, neg = np.asarray(positives), np.asarray(negatives)
    best = max(candidates, key=lambda t: (pos >= t).mean() - (neg >= t).mean())
    return float(best), float((pos >= best).mean()), float((neg >= best).mean())


def format_results_table(results: dict[str, dict]) -> str:
    """
    Tabela markdown com um resultado de `evaluate_region_technique` por linha:
    acerto real e acaso (com IC de Wilson 95%), N real, N nulo e exclusões.
    """
    lines = ["| técnica/configuração | acerto real (IC 95%) | acaso (média ± desvio) [IC Wilson 95%] | n_real | n_null | exclusões |",
             "|---|---|---|---|---|---|"]
    for name, res in results.items():
        ex = ", ".join(f"{k}={v}" for k, v in res["exclusions"].items())
        if res["n_samples"] == 0:
            lines.append(f"| {name} | N/A | N/A | 0 | 0 | {ex} |")
            continue
        r0, r1 = res["real_wilson"]
        c0, c1 = res["chance_wilson"]
        lines.append(f"| {name} | {res['real_hit_rate']*100:.1f}% [{r0*100:.1f}%, {r1*100:.1f}%] "
                     f"| {res['chance_hit_rate']*100:.1f}% ± {res['chance_hit_std']*100:.1f}% "
                     f"[{c0*100:.1f}%, {c1*100:.1f}%] | {res['n_samples']} | {res['null_valid']} | {ex} |")
    return "\n".join(lines)
