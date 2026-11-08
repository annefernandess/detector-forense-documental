"""
Feature binária de copy-move para o classificador: 1 se a técnica 1 (busca cega com verificação do resíduo
de ruído, copy_move_residuo) aprova algum par na imagem, 0 caso contrário. Os parâmetros (piso de
intensidade, k da mediana, limiar do resíduo) são os da configuração congelada em
results/copy_move_residuo_val.json; MIN_SIDE 6 e o deslocamento de +-2 px são os de copy_move_residuo. A
cobertura mínima da técnica 1 mede regiões anotadas e não entra aqui. Sequencial, sem multiprocessing.
"""

from __future__ import annotations

from copy_move_residuo import candidatos, filtrar_pares
from dataset_loader import load_gray


def feature_copy_move(gray, piso: float, k: int, limiar: float) -> int:
    """1 se algum par candidato (intensidade >= piso) tem correlação do resíduo (mediana k x k) >= limiar."""
    return int(len(filtrar_pares(candidatos(gray, ks=(k,)), piso, k, limiar)) > 0)


def feature_lote(caminhos: list, piso: float, k: int, limiar: float) -> dict:
    """caminho (str) -> feature, imagem a imagem."""
    return {str(c): feature_copy_move(load_gray(c), piso, k, limiar) for c in caminhos}
