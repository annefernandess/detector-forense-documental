"""
Loader do dataset "Find it again!" (ICDAR 2023).

IMPORTANTE: ao extrair o dataset baixado, confira o README original dele
pra ver exatamente como as imagens e anotações estão organizadas
(pode não bater 100% com o que está assumido aqui — ajuste conforme necessário).

Objetivo deste módulo: dado o caminho da pasta do dataset, retornar uma lista
de exemplos, cada um com:
    - caminho da imagem
    - label (autêntico / adulterado)
    - metadados da adulteração, se houver (tipo, região) -- útil pra discussão
      dos resultados depois (ex: "o pipeline detecta bem adulteração de valor,
      mas erra em adulteração de assinatura")

Formato real (conferido no dataset extraído):
    - {root}/{split}.txt: CSV com cabeçalho
          image,digital annotation,handwritten annotation,forged,forgery annotations
      A coluna `forgery annotations` é "0" para autênticos ou um dict Python
      (formato VIA) com as regiões adulteradas.
    - {root}/{split}/<nome>.png e {root}/{split}/<nome>.txt -- o .txt individual
      é apenas a transcrição OCR do recibo (NÃO contém info de adulteração).
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TamperRegion:
    bbox: tuple[int, int, int, int]  # (x, y, width, height) em pixels
    # Tipo de edição da região (CPI, CUT, PIX, IMI, CPO, Autre, "None").
    # Em poucas regiões o anotador marcou mais de um tipo; nesse caso
    # os tipos vêm unidos por "+" (ex: "CPI+PIX").
    modification: str | None
    entity_type: str | None  # Product, Metadata, Total/payment, Company, Other
    is_original_area: bool | None  # True = área de origem (ex: fonte do copy-paste)


@dataclass
class DocumentSample:
    image_path: Path
    is_tampered: bool
    # Tipos distintos de modificação nas regiões não-originais (sem duplicatas)
    tamper_types: list[str] = field(default_factory=list)
    regions: list[TamperRegion] = field(default_factory=list)
    software: str | None = None  # "Software used" da anotação (paint, gimp, ...)
    ocr_path: Path | None = None  # .txt individual com a transcrição OCR


def _parse_modification(raw) -> str | None:
    """'Modified area' normalmente é {'CPI': True}; no val há um caso em que
    vem como string solta ('CPI') -- usamos a string diretamente."""
    if isinstance(raw, dict):
        keys = [k for k, v in raw.items() if v]
        return "+".join(keys) if keys else None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _parse_original_area(raw) -> bool | None:
    if isinstance(raw, str):
        if raw.strip().lower() == "yes":
            return True
        if raw.strip().lower() == "no":
            return False
    return None


def _parse_annotation(raw: str) -> dict | None:
    """Retorna o dict VIA da anotação, ou None se ausente/malformada."""
    try:
        ann = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(ann, dict) or not isinstance(ann.get("regions"), list):
        return None
    return ann


def _parse_regions(ann: dict) -> list[TamperRegion]:
    regions = []
    for reg in ann["regions"]:
        shape = reg.get("shape_attributes", {})
        attrs = reg.get("region_attributes", {})
        try:
            bbox = (int(shape["x"]), int(shape["y"]),
                    int(shape["width"]), int(shape["height"]))
        except (KeyError, TypeError, ValueError):
            continue  # região sem retângulo válido
        regions.append(TamperRegion(
            bbox=bbox,
            modification=_parse_modification(attrs.get("Modified area")),
            entity_type=attrs.get("Entity type"),
            is_original_area=_parse_original_area(attrs.get("Original area")),
        ))
    return regions


def _distinct_tamper_types(regions: list[TamperRegion]) -> list[str]:
    """Tipos distintos nas regiões não-originais (is_original_area False/None).
    Tipos combinados ("CPI+PIX") são separados; "None" é ignorado."""
    types: list[str] = []
    for r in regions:
        if r.is_original_area is True or r.modification is None:
            continue
        for t in r.modification.split("+"):
            if t != "None" and t not in types:
                types.append(t)
    return types


def load_dataset(dataset_root: str, split: str, return_diagnostics: bool = False):
    """
    Carrega o split ('train', 'test' ou 'val') do Find it again!.

    Retorna list[DocumentSample]. Se `return_diagnostics=True`, retorna a tupla
    (samples, label_annotation_mismatches, ambiguous_regions):
        - label_annotation_mismatches: nomes das imagens em que `forged` diverge
          da presença de anotação (forged=1 sem anotação válida, ou forged=0
          com anotação).
        - ambiguous_regions: pares (image_path, TamperRegion) de regiões marcadas
          como área original (is_original_area=True) mas com uma modificação
          real (diferente de "None").
    Nada disso é corrigido: o rótulo segue sempre a coluna `forged` e as
    regiões são mantidas como anotadas.
    """
    if split not in ("train", "test", "val"):
        raise ValueError(f"split inválido: {split!r}")

    root = Path(dataset_root)
    samples: list[DocumentSample] = []
    mismatches: list[str] = []
    ambiguous_regions: list[tuple[Path, TamperRegion]] = []
    n_malformed = 0

    with open(root / f"{split}.txt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["image"].strip()
            image_path = root / split / name
            is_tampered = row["forged"].strip() == "1"
            raw_ann = (row.get("forgery annotations") or "").strip()
            has_ann = raw_ann not in ("", "0")

            if is_tampered != has_ann:
                mismatches.append(name)

            sample = DocumentSample(
                image_path=image_path,
                is_tampered=is_tampered,
                ocr_path=image_path.with_suffix(".txt"),
            )

            if is_tampered:
                ann = _parse_annotation(raw_ann)
                if ann is None:
                    n_malformed += 1
                else:
                    sample.regions = _parse_regions(ann)
                    sample.tamper_types = _distinct_tamper_types(sample.regions)
                    sample.software = ann.get("file_attributes", {}).get("Software used")
                    for r in sample.regions:
                        if (r.is_original_area is True and r.modification
                                and r.modification != "None"):
                            ambiguous_regions.append((image_path, r))

            samples.append(sample)

    if n_malformed:
        print(f"aviso: {n_malformed} imagens com anotação ausente/malformada "
              f"tratadas como regions=[] ({split})")
    if mismatches:
        print(f"aviso: {len(mismatches)} imagens com forged divergente da "
              f"anotação ({split}): {mismatches}")
    if ambiguous_regions:
        print(f"aviso: {len(ambiguous_regions)} regiões marcadas como área "
              f"original mas com modificação real ({split})")

    if return_diagnostics:
        return samples, mismatches, ambiguous_regions
    return samples


def load_pt_test_set(folder: str) -> list[DocumentSample]:
    """
    Loader do mini-conjunto próprio em português (fim de semana 2).
    Sugestão de organização:
        pt_test_set/
            authentic/doc1.jpg, doc2.jpg, ...
            tampered/doc1.jpg, doc2.jpg, ...
    """
    raise NotImplementedError


if __name__ == "__main__":
    from collections import Counter
    from pprint import pprint

    samples, mismatches, ambiguous = load_dataset(
        "data/findit2", "test", return_diagnostics=True)

    n_tampered = sum(s.is_tampered for s in samples)
    print(f"total: {len(samples)} | autênticos: {len(samples) - n_tampered} "
          f"| adulterados: {n_tampered}")
    print(f"label_annotation_mismatches: {len(mismatches)} {mismatches}")
    print(f"ambiguous_regions: {len(ambiguous)}")
    for path, region in ambiguous:
        print(f"  {path.name}: {region}")

    counts = Counter(t for s in samples for t in s.tamper_types)
    print("tamper_types mais comuns (nº de imagens):", counts.most_common())

    print("\nexemplo adulterado:")
    pprint(next(s for s in samples if s.is_tampered and s.regions))
