# Detector forense de adulteração em recibos

Projeto final de Processamento Digital de Imagens (PDI), UFPB, prof. Leonardo Vidal Batista.

Este repositório avalia técnicas clássicas de PDI para localizar e sinalizar adulterações em recibos digitalizados: Error Level Analysis (ELA), JPEG Ghost e detecção de cópia e colagem (copy-move) por correlação normalizada. As técnicas são medidas contra um acaso explícito (pseudo-regiões sorteadas com texto), e um classificador de regressão logística resume a imagem inteira. O classificador final é o candidato de 13 features (as 12 do v2 mais uma feature binária de copy-move com verificação do resíduo de ruído); o v2 fica como referência. Há também uma interface Streamlit de triagem, que ainda usa o v2. Os resultados são honestos e, em boa parte, modestos: a localização funciona quando a caixa adulterada é conhecida, a busca cega de copy-move sozinha não ajuda, e o ganho do classificador final sobre o v2 é pequeno e incerto. A seção [Limitações](#limitações) detalha o que cada número mede e o que não mede.

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
| **Classificador final** (v2 + `copy_move_par`), AUC | validação cruzada, train + val (769) | 0,728 ± 0,041 (v2: 0,665 ± 0,047) | | `results/classificador_copy_move_cv.json` |
| **Classificador final**, AUC | *Find it again!*, test (218; 35 adulteradas) | 0,721 [0,627; 0,810] (v2: 0,626 [0,519; 0,730]); diferença pareada +0,095 [+0,005; +0,185] | | `results/teste_final_candidato.json`, `results/configuracao_candidato.json` |
| **Classificador final**, sensibilidade / especificidade no limiar 0,3886 | previsões fora da amostra da validação cruzada, train + val | 75,3% / 60,1% | | `results/configuracao_candidato.json` |
| Feature `copy_move_par` sozinha | *Find it again!*, test | 51,4% das adulteradas e 22,4% das autênticas com par; AUC 0,645 | | `results/teste_final_candidato.json` |
| Busca de copy-move com resíduo de ruído (técnica 1), J no val | *Find it again!*, val | 0,318 (busca só por intensidade: 0,195); cobertura de regiões CPI 13,2% (antes 7,9%) | | `results/copy_move_residuo_val.json` |
| Ajuste de parâmetros (ELA, Ghost, busca cega) | desenvolvimento: CORD treino sintético (766); final: CORD sintético (95) | só o ELA trocou de configuração (p95), pela queda do acaso; o acerto real não subiu | | `results/ajuste_desenvolvimento.csv`, `results/ajuste_final.json` |

A configuração padrão de cada técnica é o resultado principal. O ajuste de parâmetros (notebook 06) não trouxe ganho real e não altera as linhas acima.

## Estrutura

```
app/            interface Streamlit de triagem (app/README.md)
data/           dados locais, não versionados (veja "Dados")
material/       slides da disciplina (não versionado)
notebooks/      00 a 10, em ordem; arquivo/ guarda o histórico do classificador
report/         fundamentação teórica e referências
results/        modelos e limiares congelados (v2 e candidato), JSON e CSV dos resultados
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
| `07_copy_move_residuo` | técnica 1: busca cega com verificação do resíduo de ruído |
| `08_ruido_fundo` | ruído local (MAD) e fundo do papel como mapa e como feature (resultado negativo como feature) |
| `09_ruido_local` | ruído local revisado (desvio-padrão, calibração) como mapa e como feature (resultado negativo como feature) |
| `10_copy_move_como_feature` | a técnica 1 como feature binária do classificador, na validação cruzada |

Os scripts `notebooks/diagnostico_*.py`, `notebooks/teste_final_candidato.py` e `notebooks/congelar_candidato.py` são diagnósticos temporários, a rodada única no test e o congelamento do candidato; cada um grava o hash do código usado.

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

- **Find it again!** (ICDAR 2023): baixar `findit2.zip` em <http://l3i-share.univ-lr.fr/2023Finditagain/index.html> e extrair em `data/findit2/` (pastas `train/`, `val/`, `test/` e os CSV de anotações). Necessário para os notebooks 00 a 04 e 07 a 10.
- **CORD v2** (Hugging Face), parquet em `data/cord/data/`: necessário para os notebooks 05 e 06. As imagens são extraídas com `extrair_parquets` de `src/conjunto_sintetico.py` para `data/cord/imagens/` (validation e test) e `data/cord/treino/` (train).
- `data/variante_sroie_kaggle/` guarda uma variante do SROIE baixada do Kaggle. Foi só inspecionada e não entra em nenhum resultado.

## Como rodar

Notebooks, da raiz do projeto:

```bash
source venv/bin/activate
jupyter notebook notebooks/
```

Executar na ordem 00 a 10. Os notebooks 05 a 10 gravam em `results/`.

Interface:

```bash
streamlit run app/streamlit_app.py
```

pytest tests/ coleta 24 testes de test_ajuste_parametros.py, test_copy_move_feature.py, test_copy_move_residuo.py, test_gerar_copy_move_sintetico.py, test_ruido_fundo.py e test_ruido_local.py. Os demais arquivos de tests/ são scripts com verificações no nível do módulo, e rodam com python tests/<arquivo>.py. Comandos:

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
- **Classificador final: ganho pequeno e incerto.** O candidato (v2 + `copy_move_par`) tem AUC 0,721 no test contra 0,626 do v2, mas a diferença pareada é +0,095 com IC [+0,005; +0,185]: o limite inferior fica quase em zero, e o test tem só 35 imagens adulteradas.
- **Test usado três vezes.** O test foi aberto na v1 e na v2 do classificador e na rodada única do candidato; a decisão de buscar uma feature de copy-move veio depois dos resultados anteriores, então os números do test são otimistas. Ver `notebooks/arquivo/HISTORICO.md`.
- **Feature sozinha fraca.** `copy_move_par` tem AUC 0,645 no test e marca 22,4% das autênticas; o ganho vem de ela ser um sinal quase independente das 12 features do v2. Letras iguais com ruído independente passam no filtro do resíduo, porque a borda do traço domina o resíduo na caixa justa.
- **Configuração da técnica 1 escolhida no val.** O piso 0,95, a mediana 3 x 3 e o limiar 0,9 foram escolhidos no val do *Find it again!*, que faz parte dos dados de treino do classificador final; a validação cruzada do candidato não é independente dessa escolha (no train, que não participou dela, a AUC isolada da feature é a mesma: 0,673 contra 0,663 no val).
- **Modelo do test não salvo.** O AUC do test foi calculado com um modelo treinado na rodada de 6/11 e não salvo. O modelo salvo nesta configuração é a reprodução da receita, com equivalência verificada por determinismo no train, e não o objeto avaliado no test.
- **No conjunto sintético, o v2 fica perto do acaso.** AUC 0,487 (intervalo [0,404; 0,570], que inclui 0,5); o candidato não foi avaliado nesse conjunto.
- **Interface.** A interface Streamlit ainda usa o v2 (`results/modelo_v2.joblib`); o candidato ainda não foi integrado.
- **Busca cega sem ganho.** Com a configuração padrão, a busca cega marca 17,5% das autênticas do *Find it again!* e cobre só 10,5% das regiões CPI. No conjunto sintético, acerta 1 de 95.
- **Conjunto sintético de copy-move.** As adulterações do CORD são colagens exatas de uma caixa de texto, geradas por programa (`src/gerar_copy_move_sintetico.py`, congelado em `results/gerador_congelado.json`). São o caso mais fácil para um método de correlação e não representam edições feitas por pessoas. Dez das 200 notas (fotos de fundo escuro) saíram das duas classes porque o gerador não achou caixa.
- **Uso duplo dos dados de avaliação.** O conjunto `data/cord_sintetico/` foi usado duas vezes: no notebook 05 e na avaliação final do notebook 06.
- **Viés do dataset.** No *Find it again!*, as imagens adulteradas tendem a estar em RGBA e as autênticas não. Todas as técnicas analisam tons de cinza para não usar esse sinal.
- **Domínio.** Resultados valem para recibos digitalizados parecidos com os do SROIE e do CORD.
- A ferramenta é apoio à triagem. Não é prova pericial.

## Referências

Martínez Tornés et al. (2023), *Find it again!*, ICDAR. Fundamentação e demais referências em [`report/`](report/).
