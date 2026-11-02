# Interface do detector

Página única, em três etapas, para quem não conhece o projeto: **1. Envie a imagem**, **2. Veja o
resultado**, **3. Entenda o porquê**. Usa o modelo e o limiar congelados da v2
(`results/modelo_v2.joblib` e `results/configuracao_congelada.json`). Não treina, não recalibra e não
lê o conjunto de teste.

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
   pontuação entre os documentos de referência (conjunto de validação) e os números medidos no teste, em
   texto: cerca de 1 em cada 2 documentos editados sinalizados, 1 em cada 5 sinalizados realmente editados
   e 1 em cada 3 documentos sem edição sinalizados. Os números são calculados das contagens
   (`METRICAS_TEST` em `src/pipeline.py`) e conferidos contra o notebook 04 por `tests/test_pipeline.py`.
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
- **O classificador é fraco.** Com AUC de 0,626, ele deixa passar mais da metade dos documentos editados
  (19 de 35) e sinaliza cerca de 1 em cada 3 documentos sem edição (64 de 183). Entre os sinalizados, só 1
  em cada 5 é editado.
- **Os números do teste são otimistas.** O conjunto de teste já havia sido usado em uma tentativa
  anterior do classificador (v1).
- **A posição entre os documentos de referência é otimista.** O modelo final foi treinado com treino e
  validação juntos, então as pontuações de referência são de dados vistos no treino. Ela precisa de
  `results/probabilidades_val_v2.csv`, que não vai para o git; sem o arquivo, a página avisa que a
  comparação não está disponível.
- **Os exemplos vêm do conjunto de dados de pesquisa** (`data/findit2/`), que não vai para o git. Sem ele,
  os exemplos não aparecem. Antes de publicar a página, é preciso conferir se a licença do conjunto de
  dados permite exibir os recibos.
- **Domínio restrito.** O modelo foi treinado em recibos do conjunto *Find it again!* (texto escuro sobre
  fundo claro). Outros documentos podem se comportar de forma imprevisível.
- **Arquivos JPG enviados já foram recomprimidos**, o que altera o mapa de diferença de compressão e o de
  histórico de compressão.
- **Acessibilidade.** O botão de envio é traduzido por CSS; leitores de tela podem ler o nome original em
  inglês. O Streamlit define o idioma da página como inglês.
