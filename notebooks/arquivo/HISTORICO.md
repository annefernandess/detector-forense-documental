# Histórico do classificador de adulteração (nível da imagem)

| versão | features | escolha | avaliação no test | AUC no test | situação |
|---|---|---|---|---|---|
| v1 (`notebooks/arquivo/04_classificador_v1_exploratorio.ipynb`) | 7 features globais | modelo e limiar escolhidos no val | test aberto pela 1ª vez | 0,545 | exploratório |
| v2 (`notebooks/04_classificador.ipynb`) | 12 features (7 globais + 5 por componente de texto) | validação cruzada no train + val | test aberto pela 2ª vez | 0,626 | **estimativa vigente** |
| v3 (`notebooks/arquivo/06_alinhamento_sem_ganho.ipynb`) | 12 features + alinhamento de linha de base | validação cruzada aninhada no train + val | critério de parada não atingido; test não aberto | — | sem ganho (AUC externa de 0,663 contra 0,665 sem o alinhamento) |

O conjunto de teste foi usado em duas avaliações do classificador: na v1, com AUC de 0,545, e na v2, com AUC de 0,626. A v3 não chegou ao teste, porque o alinhamento não superou em 0,03 a AUC da validação cruzada da v2. A estimativa final é a da v2 (AUC de 0,626 no teste), com a ressalva de que ela é otimista: o teste já tinha sido visto na v1, e a decisão de buscar novas features foi tomada depois desse primeiro resultado.
