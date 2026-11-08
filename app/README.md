# Interface do detector

Página única, em três etapas, para quem não conhece o projeto: **1. Envie a imagem**, **2. Veja o
resultado**, **3. Entenda o porquê**. Usa o classificador candidato congelado (v2 + `copy_move_par`, 13
features) e o limiar 0,38857 (`results/modelo_candidato.joblib` e `results/configuracao_candidato.json`); o
hash do modelo é conferido antes do uso e, se não bater, a análise para. O `copy_move_par` vem da técnica 1
(`src/copy_move_residuo.py`, configuração congelada). Não treina, não recalibra e não lê o conjunto de teste.
O modelo v2 (`results/modelo_v2.joblib`) continua no repositório como referência.

## Como rodar

Na raiz do projeto:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

O tema claro fixo vem de `.streamlit/config.toml` (na raiz). O modelo foi salvo com a versão 1.6.1 do
scikit-learn; versões diferentes podem recusar o arquivo. Imagens grandes (digitalizações de cerca de
35 milhões de pixels) levam até cerca de 10 segundos.

## O que a tela mostra

1. **Envie a imagem.** Botão de envio (PNG ou JPG, até 20 MB) e seis exemplos do conjunto de dados,
   rotulados "Exemplo". Arquivo de outro formato, grande demais ou ilegível recebe uma mensagem clara.
2. **Veja o resultado.** A faixa (*Revisar* ou *Sem sinal relevante*, com ícone e texto, nunca só cor),
   uma frase de resumo em linguagem simples e o bloco "Como ler o resultado": o que cada faixa quer dizer,
   a frase "Esta ferramenta ordena documentos para revisão. Ela não decide se houve fraude.", a posição da
   pontuação ("Esta imagem está em posição mais alta que X% dos recibos de referência deste tipo.") e os
   números medidos, em texto: cerca de 75% dos documentos editados sinalizados e cerca de 40% dos documentos
   sem edição sinalizados, nas previsões fora da amostra da validação cruzada de train + val
   (`metricas_candidato` em `src/pipeline.py`, de `results/configuracao_candidato.json`). A página mostra a
   faixa, nunca uma probabilidade. No topo da página, a declaração: "Esta ferramenta foi validada apenas para
   recibos semelhantes aos do conjunto Find it again!. Ela ordena documentos para revisão e não confirma
   fraude."
3. **Entenda o porquê.** Três mapas desenhados sobre a imagem, cada um com título, legenda de cores em uma
   linha, controle de transparência (padrão 50%), uma frase sobre o que mostra, uma sobre o que não mostra e
   um texto de ajuda ("O que é isto?"): *Mapa de diferença de compressão* (ELA), *Mapa de histórico de
   compressão* (JPEG Ghost) e *Possíveis trechos repetidos* (copy-move). Os nomes técnicos só aparecem
   dentro dos textos de ajuda. Sem histórico JPEG, o segundo mapa explica por que não se aplica.

Abaixo, "O que esta ferramenta não faz" (três itens). As caixas automáticas de região (`src/regioes.py`)
não aparecem mais na tela; o módulo, os testes e `results/regioes_*` continuam no repositório.

## Modo demonstração (para apresentações)

Link discreto abaixo dos exemplos (`?demo=1`). Mostra um aviso no topo e, **só nos exemplos do conjunto de
dados**, as caixas do gabarito em vermelho. Uma imagem enviada pelo usuário nunca mostra gabarito. O
parâmetro `?exemplo=<arquivo>` seleciona um exemplo pelo link.

## Textos e estilo

- `app/textos.py`: todas as frases da interface, para revisão. `tests/test_app_textos.py` verifica que não
  há jargão fora dos textos de ajuda (que têm duas frases cada) e que "fraude", "adulterada" e
  "falsificada" não aparecem como afirmação, tanto nas frases quanto na página renderizada.
- `app/estilo.py`: paleta em tokens `:root` (texto de pelo menos 6,7:1 de contraste no navegador),
  etapas, faixa de resultado e figuras com texto alternativo.

## Limitações

- **A ferramenta ordena documentos para revisão; não decide se houve fraude.**
- **O classificador é fraco.** No teste, a AUC do candidato é 0,721 (a do v2, 0,626), com ganho pequeno e
  incerto (diferença +0,095, intervalo [+0,005; +0,185]). No limiar congelado, a ferramenta sinaliza cerca de
  40% dos documentos sem edição. Ver o README do projeto.
- **O teste foi usado três vezes** (v1, v2 e a rodada única do candidato), então os números do teste são
  otimistas. Os números mostrados na tela vêm da validação cruzada de train + val, não do teste, e também são
  otimistas: os mesmos recibos ajudaram a escolher o limite de atenção.
- **A posição é calculada sobre previsões fora da amostra.** A referência (`results/referencia_percentil_candidato.json`)
  tem 2307 previsões da validação cruzada 5 x 3 (seed 42) de train + val, com o hash conferido na carga, e vai
  para o git porque não lista imagens. Se o arquivo faltar, a página avisa que a comparação não está disponível.
- **Os pares de trechos repetidos vêm da técnica 1.** O limite de "muitos pares" do texto explicativo
  (`MUITOS_PARES` = 15 em `src/explicacao.py`) é o percentil 99 dos pares aprovados por imagem nas autênticas
  de train + val (`app/calibrar_limite_pares.py`, `results/regioes_config.json`).
- **Os exemplos vêm do conjunto de dados de pesquisa** (`data/findit2/`), que não vai para o git. Sem ele,
  os exemplos não aparecem. Antes de publicar a página, é preciso conferir se a licença do conjunto de
  dados permite exibir os recibos.
- **Domínio restrito.** O modelo foi treinado em recibos do conjunto *Find it again!* (texto escuro sobre
  fundo claro). Outros documentos podem se comportar de forma imprevisível.
- **Arquivos JPG enviados já foram recomprimidos**, o que altera o mapa de diferença de compressão e o de
  histórico de compressão.
- **Acessibilidade.** O botão de envio é traduzido por CSS; leitores de tela podem ler o nome original em
  inglês. O Streamlit define o idioma da página como inglês.
