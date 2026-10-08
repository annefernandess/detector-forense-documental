# Fundamentação teórica — Error Level Analysis (ELA)

Ideia: recomprimir a imagem em JPEG numa qualidade conhecida e comparar
com a original. Regiões editadas tendem a ter um "nível de erro" diferente
do resto da imagem, porque foram salvas com histórico de compressão distinto.

Referência de partida: Krawetz, N. "A Picture's Worth: Digital Image Analysis
and Forensics" (2007).


## Fundamentação teórica

### 1. Compressão JPEG e a DCT

O JPEG (modo baseline) não armazena os pixels diretamente. A imagem é
convertida para o espaço YCbCr (luminância Y e duas crominâncias, que
normalmente são subamostradas), e cada canal é dividido em blocos de 8x8
pixels. Em cada bloco f(x, y), deslocado para o intervalo [-128, 127],
aplica-se a Transformada Discreta de Cosseno bidimensional (DCT-II):

```
F(u, v) = 1/4 · C(u) · C(v) · Σ_{x=0..7} Σ_{y=0..7} f(x, y)
          · cos[(2x + 1)uπ / 16] · cos[(2y + 1)vπ / 16]

com C(0) = 1/√2 e C(k) = 1 para k > 0.
```

A DCT em si é inversível (sem perda): ela apenas reescreve o bloco como uma
combinação de 64 funções-base cossenoidais. O coeficiente F(0, 0) (DC) é a
média do bloco, e os coeficientes com (u, v) maiores representam
frequências espaciais cada vez mais altas (bordas e texturas finas).
Em imagens naturais, a energia fica concentrada nas baixas frequências.

A perda de informação acontece na QUANTIZAÇÃO. Cada coeficiente é dividido
por um valor de uma tabela Q(u, v) e arredondado:

```
F_q(u, v) = round( F(u, v) / Q(u, v) )
```

A tabela é escalada pelo fator de qualidade (1-100): qualidade menor implica
passos Q(u, v) maiores, sobretudo nas altas frequências, às quais a visão
humana é menos sensível. Na descompressão, faz-se F_q · Q, aplica-se a DCT
inversa e os valores são arredondados e limitados a [0, 255]. Por isso, após
um ciclo JPEG, os coeficientes DCT de cada bloco ficam (aproximadamente)
"presos" a múltiplos inteiros de Q(u, v).

### 2. Por que recomprimir revela edições

Se uma imagem que já passou por JPEG for recomprimida com a mesma tabela de
quantização (ou uma parecida), os coeficientes dos blocos não editados já
estão sobre a "grade" de múltiplos de Q(u, v): a nova quantização quase não
os altera. O erro de recompressão nesses blocos é pequeno, porque eles já
"convergiram" para aquele padrão. Cada nova recompressão perde cada vez
menos informação.

Uma região editada (um trecho colado de outra imagem, um número redesenhado
no Paint/GIMP, texto inserido digitalmente) nunca passou por essa mesma
quantização, ou passou por uma quantização diferente e com outro alinhamento
da grade 8x8. Seus coeficientes DCT não estão sobre a grade e, ao
recomprimir, sofrem um arredondamento maior. No domínio dos pixels, isso
aparece como uma diferença maior entre a imagem e a sua versão recomprimida.

A ELA calcula esse "nível de erro" pixel a pixel:

```
E(x, y) = | I(x, y) - JPEG_q( I )(x, y) |
```

onde JPEG_q(I) é a imagem comprimida e descomprimida com qualidade q. Como
E costuma ter valores muito pequenos (poucos níveis de cinza), ela é
multiplicada por um fator de amplificação para ser visualizada como mapa de
calor. Regiões com erro destoante do entorno são candidatas a edição.

### 3. Limitações (importantes para documentos)

- Bordas de alto contraste, como o texto preto sobre o papel branco de um
  recibo, concentram energia em altas frequências, que são as mais
  quantizadas. Por isso o texto inteiro "acende" no mapa de ELA mesmo sem
  edição, e um caractere adulterado precisa se destacar em relação aos
  caracteres vizinhos, não ao fundo.
- A ELA pressupõe que a imagem tenha histórico JPEG. No dataset
  "Find it again!" as imagens estão em PNG (sem perda), então a
  recompressão aqui é o PRIMEIRO ciclo JPEG que o arquivo PNG sofre. Algum
  sinal só aparece se a imagem de origem já carregava os vestígios de um
  JPEG anterior (por exemplo, na digitalização) e a edição posterior
  quebrou esses vestígios localmente.
- Se a qualidade usada na recompressão estiver muito distante da qualidade
  original, até os blocos intactos apresentam erro alto, e o contraste
  entre região editada e não editada diminui. Por isso a qualidade é um
  parâmetro a ser calibrado.
- Regiões lisas (fundo branco uniforme) quase não têm erro em nenhum caso,
  então uma edição que só apague conteúdo (pintar de branco) tende a ser
  pouco visível.
