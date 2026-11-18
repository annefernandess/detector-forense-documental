# Histórico do classificador de adulteração (nível da imagem)

| versão | features | escolha | avaliação no test | AUC no test | situação |
|---|---|---|---|---|---|
| v1 (`notebooks/arquivo/04_classificador_v1_exploratorio.ipynb`) | 7 features globais | modelo e limiar escolhidos no val | test aberto pela 1ª vez | 0,545 | exploratório |
| v2 (`notebooks/04_classificador.ipynb`) | 12 features (7 globais + 5 por componente de texto) | validação cruzada no train + val | test aberto pela 2ª vez | 0,626 | referência |
| v3 (`notebooks/arquivo/06_alinhamento_sem_ganho.ipynb`) | 12 features + alinhamento de linha de base | validação cruzada aninhada no train + val | critério de parada não atingido; test não aberto | — | sem ganho (AUC externa de 0,663 contra 0,665 sem o alinhamento) |
| candidato (`notebooks/10_copy_move_como_feature.ipynb`, `notebooks/teste_final_candidato.py`) | 12 features do v2 + `copy_move_par` | validação cruzada no train + val, com critério fixado antes | test aberto pela 3ª vez, em rodada única | 0,721 | **classificador final** |

O conjunto de teste foi usado em três avaliações do classificador: na v1, com AUC de 0,545; na v2, com AUC de 0,626; e na rodada única do candidato, com AUC de 0,721 contra 0,626 do v2 (diferença pareada de +0,095, IC 95% [+0,005; +0,185]; `results/teste_final_candidato.json`). A v3 não chegou ao teste, porque o alinhamento não superou em 0,03 a AUC da validação cruzada da v2. O classificador final é o candidato, com a ressalva de que os números do teste são otimistas: o teste já tinha sido visto nas versões anteriores, e a decisão de buscar novas features foi tomada depois desses resultados.
