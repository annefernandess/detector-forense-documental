# Detector forense de adulteração em recibos

Projeto final de Processamento Digital de Imagens (PDI), UFPB, prof. Leonardo Vidal Batista.

Este repositório avalia técnicas clássicas de PDI para localizar e sinalizar adulterações em recibos digitalizados: Error Level Analysis (ELA), JPEG Ghost e detecção de cópia e colagem (copy-move) por correlação normalizada. As técnicas são medidas contra um acaso explícito (pseudo-regiões sorteadas com texto), e um classificador de regressão logística sobre 12 features resume a imagem inteira. Há também uma interface Streamlit de triagem. Os resultados são honestos e, em boa parte, modestos: a localização funciona quando a caixa adulterada é conhecida, mas o classificador fica perto do acaso e a busca cega de copy-move não ajuda. A seção [Limitações](#limitações) detalha o que cada número mede e o que não mede.

## Resultados principais

Cada linha cita o arquivo de onde o número vem. "Acaso" é a taxa de acerto de pseudo-regiões com texto, do mesmo tamanho e sem sobreposição com a região anotada. Intervalos de Wilson de 95%, salvo indicação.

| Medida | Conjunto | Resultado | Acaso | Fonte |
|---|---|---|---|---|
| ELA, localização (caixa conhecida) | *Find it again!*, test (n=35) | 97,1% (34/35) | 51,1% ± 10,3% | `notebooks/01_ela.ipynb` |
| ELA, localização (caixa conhecida) | CORD sintético (n=95) | 89,5% [81,7; 94,2] | 45,0% | `results/generalizacao_sintetica.json` |
| JPEG Ghost, localização (16x16) | *Find it again!*, test (n=34) | 50,0% [34,1; 65,9] | 11,6% ± 12,9% | `notebooks/02_jpeg_ghost.ipynb` |
| Copy-move, caixa conhecida | *Find it again!*, test, imagens com CPI (n=27) | 81,5% [63,3; 91,8] | 10,1% ± 12,8% | `notebooks/03_copy_move.ipynb` |
| Copy-move, caixa conhecida | CORD sintético (n=93) | 100% [96,0; 100] | 11,7% ± 12,0% | `results/generalizacao_sintetica.json` |
| Copy-move, busca cega (limiar 0,99) | *Find it again!*, test | 33,3% (9/27) das imagens com CPI; 17,5% (32/183) das autênticas | não se aplica | `notebooks/03_copy_move.ipynb` |
| Copy-move, busca cega (limiar 0,99, cobertura 0,5) | CORD sintético (n=95) | 1,1% [0,2; 5,7] | 0,0% | `results/generalizacao_sintetica.json` |
| Classificador v2, AUC | validação cruzada, train + val | 0,665 ± 0,047 | | `results/configuracao_congelada.json` |
| Classificador v2, sensibilidade / especificidade / AUC | *Find it again!*, test | 45,7% [30,5; 61,8] / 65,0% [57,9; 71,6] / 0,626 | | `notebooks/04_classificador.ipynb` |
| Classificador v2, sensibilidade / falso alarme / AUC | CORD sintético (95 + 95) | 32,6% [24,0; 42,6] / 26,3% [18,5; 36,0] / 0,487 [0,404; 0,570] | | `results/generalizacao_sintetica.json` |
| Ajuste de parâmetros (ELA, Ghost, busca cega) | desenvolvimento: CORD treino sintético (766); final: CORD sintético (95) | só o ELA trocou de configuração (p95), pela queda do acaso; o acerto real não subiu | | `results/ajuste_desenvolvimento.csv`, `results/ajuste_final.json` |

A configuração padrão de cada técnica é o resultado principal. O ajuste de parâmetros (notebook 06) não trouxe ganho real e não altera as linhas acima.

## Estrutura

```
app/            interface Streamlit de triagem (app/README.md)
data/           dados locais, não versionados (veja "Dados")
material/       slides da disciplina (não versionado)
notebooks/      00 a 06, em ordem; arquivo/ guarda o histórico do classificador
report/         fundamentação teórica e referências
results/        modelo e limiar congelados, JSON e CSV dos resultados
src/            toda a lógica (técnicas, avaliação, classificador, gerador sintético)
tests/          testes com assert
.streamlit/     tema da interface
```

Notebooks:

| Notebook | Conteúdo |
|---|---|
| `00_explorar_dataset` | anotações e viés do *Find it again!* |
| `01_ela` | ELA: varredura de qualidades e localização |
| `02_jpeg_ghost` | JPEG Ghost: mapa de qualidade por bloco e localização |
| `03_copy_move` | copy-move com caixa conhecida e busca cega |
| `04_classificador` | classificador de 12 features, validação cruzada e test |
| `05_generalizacao_sintetica` | modelo congelado em notas CORD com copy-move sintético |
| `06_ajuste_parametros` | grade de parâmetros das técnicas, desenvolvimento e avaliação final |

Os JSON em `results/` que citam "notebook 07" e "notebook 08" se referem aos notebooks 05 e 06, renomeados depois de gerados.

## Instalação

Python 3.9.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dados

Nada em `data/` vai para o Git.

- **Find it again!** (ICDAR 2023): baixar `findit2.zip` em <http://l3i-share.univ-lr.fr/2023Finditagain/index.html> e extrair em `data/findit2/` (pastas `train/`, `val/`, `test/` e os CSV de anotações). Necessário para os notebooks 00 a 04.
- **CORD v2** (Hugging Face), parquet em `data/cord/data/`: necessário para os notebooks 05 e 06. As imagens são extraídas com `extrair_parquets` de `src/conjunto_sintetico.py` para `data/cord/imagens/` (validation e test) e `data/cord/treino/` (train).
- `data/variante_sroie_kaggle/` guarda uma variante do SROIE baixada do Kaggle. Foi só inspecionada e não entra em nenhum resultado.

## Como rodar

Notebooks, da raiz do projeto:

```bash
source venv/bin/activate
jupyter notebook notebooks/
```

Executar na ordem 00 a 06. Os notebooks 05 e 06 gravam em `results/`.

Interface:

```bash
streamlit run app/streamlit_app.py
```

pytest tests/ coleta 5 testes de test_ajuste_parametros.py e test_gerar_copy_move_sintetico.py. Os demais arquivos de tests/ são scripts com verificações no nível do módulo, e rodam com python tests/<arquivo>.py. Comandos:

```bash
python tests/test_app_textos.py
python tests/test_classification.py
python tests/test_copy_move.py
python tests/test_explicacao.py
python tests/test_pipeline.py
python tests/test_regioes.py
```

## Limitações

- **Localização com caixa conhecida.** As linhas de ELA, Ghost e copy-move com caixa conhecida medem se a região anotada se destaca de um controle com texto. Não medem se a técnica encontra a região sozinha.
- **Classificador próximo do acaso.** A AUC no test é 0,626, e no conjunto sintético é 0,487 (intervalo [0,404; 0,570], que inclui 0,5). O test foi usado em duas avaliações do classificador (v1 e v2), então 0,626 é otimista. Ver `notebooks/arquivo/HISTORICO.md`.
- **Busca cega sem ganho.** Com a configuração padrão, a busca cega marca 17,5% das autênticas do *Find it again!* e cobre só 10,5% das regiões CPI. No conjunto sintético, acerta 1 de 95.
- **Conjunto sintético de copy-move.** As adulterações do CORD são colagens exatas de uma caixa de texto, geradas por programa (`src/gerar_copy_move_sintetico.py`, congelado em `results/gerador_congelado.json`). São o caso mais fácil para um método de correlação e não representam edições feitas por pessoas. Dez das 200 notas (fotos de fundo escuro) saíram das duas classes porque o gerador não achou caixa.
- **Uso duplo dos dados de avaliação.** O conjunto `data/cord_sintetico/` foi usado duas vezes: no notebook 05 e na avaliação final do notebook 06.
- **Viés do dataset.** No *Find it again!*, as imagens adulteradas tendem a estar em RGBA e as autênticas não. Todas as técnicas analisam tons de cinza para não usar esse sinal.
- **Domínio.** Resultados valem para recibos digitalizados parecidos com os do SROIE e do CORD.
- A ferramenta é apoio à triagem. Não é prova pericial.

## Referências

Martínez Tornés et al. (2023), *Find it again!*, ICDAR. Fundamentação e demais referências em [`report/`](report/).
