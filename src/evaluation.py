from __future__ import annotations

import random
import statistics
from dataclasses import replace

import cv2
import numpy as np
from PIL import Image

from dataset_loader import DocumentSample, TamperRegion

def get_text_mask(image_path: str) -> np.ndarray | None:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, text_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(text_mask) > text_mask.size / 2:
        text_mask = cv2.bitwise_not(text_mask)
    return text_mask

def _generate_pseudo_region(
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
    Generaliza a avaliação de uma técnica de detecção local em imagens adulteradas,
    comparando os acertos nas regiões reais contra uma linha de base nula (acaso).
    """
    rng = random.Random(seed)
    random.seed(seed)

    valid_samples = [s for s in samples if s.is_tampered and s.regions]

    n_valid = 0
    real_hits_total = 0
    trial_hits = [0] * chance_trials

    for s in valid_samples:
        text_mask = get_text_mask(s.image_path)
        if text_mask is None:
            continue

        # 1) Avaliação na região real
        s_in, s_out = score_fn(s.image_path, s.regions)
        if s_in is None or s_out is None:
            continue
            
        real_hit = hit_fn(s_in, s_out)

        # 2) Linha de base nula (pseudo-regiões)
        mods = [r for r in s.regions if getattr(r, "is_original_area", None) is not True] or s.regions
        ref = max(mods, key=lambda r: r.bbox[2] * r.bbox[3])

        with Image.open(s.image_path) as img:
            img_width, img_height = img.size

        masked_regions = [replace(r, is_original_area=True) for r in s.regions]

        trial_results = []
        all_trials_successful = True

        for _ in range(chance_trials):
            pseudo = _generate_pseudo_region(
                img_width, img_height, ref.bbox[2], ref.bbox[3], s.regions, text_mask, rng
            )
            if not pseudo:
                all_trials_successful = False
                break
                
            ps_in, ps_out = score_fn(s.image_path, [pseudo] + masked_regions)
            if ps_in is None or ps_out is None:
                all_trials_successful = False
                break
                
            trial_results.append(hit_fn(ps_in, ps_out))

        if all_trials_successful:
            n_valid += 1
            if real_hit:
                real_hits_total += 1
            for i, hit in enumerate(trial_results):
                if hit:
                    trial_hits[i] += 1

    if n_valid == 0:
        return {
            "n_samples": 0,
            "real_hit_rate": 0.0,
            "chance_hit_rate": 0.0,
            "chance_hit_std": 0.0,
            "real_valid": 0,
            "null_valid": 0,
            "real_hits": 0,
        }

    real_rate = real_hits_total / n_valid
    trial_rates = [th / n_valid for th in trial_hits]
    chance_mean = statistics.mean(trial_rates) if chance_trials > 0 else 0.0
    chance_std = statistics.stdev(trial_rates) if chance_trials > 1 else 0.0

    return {
        "n_samples": n_valid,
        "real_hit_rate": real_rate,
        "chance_hit_rate": chance_mean,
        "chance_hit_std": chance_std,
        "real_valid": n_valid,
        "null_valid": n_valid * chance_trials,
        "real_hits": real_hits_total,
    }
