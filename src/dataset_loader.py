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
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentSample:
    image_path: Path
    is_tampered: bool
    tamper_type: str | None = None  # ex: "valor", "data", "assinatura"
    bbox: tuple | None = None  # região adulterada, se anotada


def load_dataset(dataset_root: str) -> list[DocumentSample]:
    """
    TODO:
        - percorrer as pastas/arquivos de anotação do dataset
        - popular uma lista de DocumentSample
        - conferir se o dataset já vem com split de treino/teste definido
          (se sim, é melhor manter o mesmo split pra comparar com os
          baselines do paper original)
    """
    raise NotImplementedError(
        "Ajustar conforme a estrutura real do dataset após extrair o zip."
    )


def load_pt_test_set(folder: str) -> list[DocumentSample]:
    """
    Loader do mini-conjunto próprio em português (fim de semana 2).
    Sugestão de organização:
        pt_test_set/
            authentic/doc1.jpg, doc2.jpg, ...
            tampered/doc1.jpg, doc2.jpg, ...
    """
    raise NotImplementedError
