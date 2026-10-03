# Detector Forense de Adulteração em Documentos

Projeto Final — Processamento Digital de Imagens (PDI) — UFPB
Entrega: 09/11

## Ideia
Detectar regiões adulteradas em documentos digitalizados (notas fiscais, boletos, contratos)
combinando técnicas clássicas de PDI: ELA, JPEG Ghost, análise de ruído local,
copy-move detection e consistência de traço/texto.

## Estrutura do projeto
```
data/
  authentic/      -> documentos originais do dataset "Find it again!"
  tampered/        -> documentos adulterados do dataset "Find it again!"
  pt_test_set/      -> mini-conjunto próprio em português (autêntico + adulterado), para teste de generalização
src/
  ela.py            -> Error Level Analysis
  jpeg_ghost.py       -> detecção de múltiplas compressões
  noise_map.py        -> mapa de ruído local
  copy_move.py        -> detecção de cópia interna (ORB)
  text_consistency.py    -> consistência de traço/alinhamento (ângulo de novidade)
  dataset_loader.py     -> leitura das anotações do dataset
  pipeline.py         -> combina as técnicas num score final
notebooks/          -> exploração e testes rápidos
report/            -> rascunho do relatório (introdução, fundamentação, etc.)
```

## Como baixar o dataset "Find it again!"
1. Acessar: http://l3i-share.univ-lr.fr/2023Finditagain/index.html
2. Baixar o zip: http://l3i-share.univ-lr.fr/2023Finditagain/findit2.zip
3. Extrair o conteúdo dentro de `data/` (deve vir com pastas de imagens autênticas/adulteradas e anotações
   — ao extrair, confira o README do próprio dataset pra saber exatamente como estão organizadas
   as pastas e o formato das anotações, e ajuste `dataset_loader.py` de acordo).

## Setup do ambiente
```bash
python3 -m venv venv
source venv/bin/activate   # no Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Checklist de hoje (dia 1)
- [ ] Baixar e extrair o dataset em `data/`
- [ ] Criar o ambiente virtual e instalar as dependências
- [ ] Abrir 5-10 imagens (autênticas e adulteradas) e olhar visualmente — sem código ainda,
      só pra entender o tipo de adulteração presente no dataset
- [ ] Rodar `notebooks/00_explorar_dataset.ipynb` pra confirmar que consegue carregar as imagens e anotações
- [ ] Separar 5 referências bibliográficas iniciais em `report/referencias.md` (ver seção abaixo)

## Referências iniciais para a fundamentação teórica
- Martínez Tornés, B. et al. "Receipt Dataset for Document Forgery Detection", ICDAR 2023.
  (paper do dataset "Find it again!" — descreve os baselines usados por eles, boa base de comparação)
- Qu, C. et al. "DocTamper" (2023) — dataset e método para localização de adulteração em documentos.
- Buscar também: artigos originais sobre "Error Level Analysis" (Krawetz, 2007 — trabalho de referência,
  não é paper acadêmico formal, mas é a origem da técnica) e sobre "JPEG Ghosts" (Farid, 2009).
- Um survey geral de "image forgery detection" ajuda a estruturar a fundamentação teórica.
