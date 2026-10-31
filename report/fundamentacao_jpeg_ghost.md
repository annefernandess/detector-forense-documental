# Fundamentação teórica — JPEG Ghost

Ideia: recomprimir a imagem em JPEG em várias qualidades e, para cada região,
ver em qual qualidade o erro de recompressão é mínimo. Esse mínimo denuncia a
qualidade com que aquela região foi comprimida pela última vez. Se uma região
"prefere" uma qualidade diferente da do resto da imagem, ela provavelmente
veio de outra fonte. Essa incompatibilidade espacial é o "fantasma" (ghost).

Referência de partida: Farid, H. "Exposing Digital Forgeries from JPEG Ghosts".
IEEE Transactions on Information Forensics and Security, 4(1), 2009.


## Fundamentação teórica

### 1. Relação com a ELA: a mesma quantização, usada de outro jeito

A base é a mesma da ELA (ver `fundamentacao_ela.md`): o JPEG divide a imagem
em blocos 8x8, aplica a DCT em cada bloco e quantiza os coeficientes:

```
F_q(u, v) = round( F(u, v) / Q_q(u, v) )
```

onde a tabela Q_q depende do fator de qualidade q. Depois de um ciclo JPEG com
qualidade q0, os coeficientes de cada bloco ficam (aproximadamente) sobre a
"grade" de múltiplos de Q_q0(u, v).

A ELA recomprime numa **única** qualidade e olha a **intensidade** do erro.
O JPEG Ghost recomprime em **várias** qualidades e olha **onde fica o mínimo**
da curva de erro. Por isso ele não depende de escolher bem uma qualidade, e
consegue dizer não só "esta região é diferente" mas também "esta região foi
comprimida em torno de q = X".

### 2. Por que o erro tem um mínimo em q0

Considera-se um coeficiente c1 que já foi quantizado com passo Δ1 (ou seja,
c1 = k · Δ1 para algum inteiro k). Recomprime-se com passo Δ2 e mede-se a
diferença ao quadrado. Farid mostra que esse erro:

- é **zero** quando Δ2 = Δ1: c1 já é múltiplo de Δ1, então re-quantizar com o
  mesmo passo não muda nada (a imagem "convergiu" para aquela grade);
- **cresce** de forma geral conforme Δ2 aumenta (passos maiores descartam
  mais informação);
- mas tem **mínimos locais** em Δ2 = Δ1 e nos divisores de Δ1, mesmo quando o
  erro "de fundo" é maior.

O passo Δ não é o fator de qualidade q: no JPEG, q (1 a 100) escala a tabela
inteira, e qualidade menor significa passos maiores. Em termos de fator de
qualidade, recomprimir com q = q0 dá um erro muito
menor que com as qualidades vizinhas (q0 - 5 ou q0 + 5). A curva

```
d(q) = (1/N) · Σ_{x,y ∈ bloco} [ I(x, y) - JPEG_q( I )(x, y) ]²
```

apresenta um **vale** em q0. Na prática o erro não é exatamente zero (os
pixels são arredondados para inteiros e limitados a [0, 255] na descompressão,
e a imagem pode ter sofrido outras operações), mas o vale continua visível.

### 3. De onde vem o "fantasma"

Supõe-se um documento comprimido originalmente com qualidade q0 = 75, no qual
o falsificador cola um trecho vindo de outra imagem que tinha sido comprimida
com qualidade q1 = 60 (ou que nunca tinha sido comprimida). Ao salvar o
resultado:

- os blocos originais continuam com a "memória" de q0 = 75: suas curvas d(q)
  têm vale em 75;
- os blocos colados têm a memória de q1 = 60 (ou nenhuma): suas curvas têm
  vale em 60, ou nenhum vale.

Calculando d(q) **por bloco** (em vez da imagem inteira) e anotando em cada
bloco o q que minimiza o erro, obtém-se um **mapa espacial de qualidades
preferidas**. Em teoria, numa imagem íntegra o mapa é quase uniforme (quase tudo
vota q0). Na prática, neste dataset, muitos blocos de imagens íntegras votam no
topo da faixa (ver limitações). Numa imagem adulterada aparece uma "mancha" que vota em outra
qualidade: o fantasma, que aparece no lugar exato da colagem.

Esse método é especialmente sensível quando q1 < q0. Se o trecho colado já
tinha sido comprimido com uma qualidade mais baixa, a perda grosseira
daquela compressão não é desfeita pela compressão posterior em q0, e o vale
em q1 sobrevive.

### 4. Escolhas de implementação

- **Blocos de 16x16 (múltiplos de 8).** O erro é agregado por bloco para
  reduzir ruído. Usar múltiplos de 8, alinhados à origem, respeita a grade de
  compressão do JPEG.
- **Faixa de qualidades 50–90, de 5 em 5.** A qualidade 95 é excluída de
  propósito. Recomprimir em 95 é quase sem perda: o erro é pequeno para
  QUALQUER bloco, com ou sem histórico, e o 95 "venceria" o argmin na maior
  parte da imagem sem informar nada sobre q0. Em um teste exploratório com
  20 imagens autênticas do test (42.737 blocos com textura, fora dos
  notebooks), incluir o 95 fazia 96,2% dos blocos votarem nele. Sem o 95, o
  mesmo efeito se repete no novo topo da faixa: 65,5% dos blocos votam em
  q = 90 (notebook `02_jpeg_ghost.ipynb`, 20 autênticas, 49.433 blocos).
  Votar na maior qualidade da faixa significa "sem vale detectável no bloco",
  e não uma qualidade de origem.
- **Blocos lisos são descartados (valor -1).** No papel em branco, d(q) ≈ 0
  para todas as qualidades e o argmin é arbitrário. Só blocos com desvio
  padrão ≥ 10 níveis de cinza (em geral, os que têm texto) votam.
- **Análise em tons de cinza**, com canal alfa composto sobre fundo branco,
  pelo mesmo motivo da ELA: evitar que o modo de cor do arquivo (L × RGBA)
  vire um falso sinal.

### 5. Limitações (importantes para documentos)

- **Imagens que nunca foram JPEG.** Esta é a limitação central para o dataset
  "Find it again!", que é distribuído em PNG (sem perda). O JPEG Ghost não
  "lê" o formato do arquivo; ele procura nos pixels o **vestígio** de uma
  compressão anterior. Há dois cenários possíveis:
    1. *A imagem nunca passou por JPEG* (ex.: renderizada direto de um PDF, ou
       capturada e salva sem perda). Nesse caso não existe grade de
       quantização para "lembrar", e a curva d(q) só decresce conforme q
       aumenta, sem vale. O argmin vai sempre para a maior qualidade testada,
       o mapa fica uniforme e a técnica **não tem o que detectar**. Ela é cega,
       mas não gera falso positivo por isso.
    2. *A imagem passou por JPEG antes de virar PNG* (ex.: foto/scan salvo
       em JPEG e depois convertido). A conversão para PNG preserva os pixels
       exatamente, incluindo a "memória" da quantização. O ghost funciona
       normalmente.

  Por isso, antes da análise, verifica-se imagem a imagem se a curva d(q) da
  imagem inteira tem um vale interno (`has_jpeg_history`). As imagens deste
  dataset vêm do SROIE e, segundo Martínez Tornés et al. (2023), são imagens
  JPEG (41 matrizes de quantização distintas). Na prática, 188 das 218 imagens
  do test (86%) apresentam um vale, concentrado em q = 70–75, ou seja, caem no
  cenário 2. O notebook `02_jpeg_ghost.ipynb` reporta essa contagem.
- **Edições que não trazem histórico próprio.** Se o falsificador não colou
  nada, mas desenhou um número no Paint ou digitou texto novo, os pixels
  editados nunca foram comprimidos. Eles não "votam" em uma qualidade baixa:
  tendem a votar na maior qualidade da faixa (curva monotônica). Isso difere
  do resto da imagem (q0) apenas quando o restante vota em outra qualidade. No
  test, 33 das 34 regiões anotadas avaliadas votam em q = 90, e o Ghost acerta
  17 de 18 quando o restante da imagem tem vale local e 0 de 16 quando não tem
  (notebook `02_jpeg_ghost.ipynb`). Votar em q = 90 não indica edição por si só,
  porque esse também é o voto mais frequente dos blocos de imagens autênticas
  (notebook `02_jpeg_ghost.ipynb`).
- **Copy-move dentro do mesmo documento.** Se o trecho colado vem do **mesmo**
  recibo, ele tem o mesmo histórico q0 do resto, e o ghost não o distingue
  (a menos que a colagem desalinhe a grade 8x8). Para esse caso, uma técnica
  específica de copy-move é mais adequada.
- **Regiões pequenas.** As adulterações em recibos costumam ser poucos
  caracteres (às vezes uma região de ~30x40 px). Com blocos de 16x16, isso
  dá só 2 a 6 blocos para votar, e a moda dentro da região fica ruidosa.
  Reduzir o bloco para 8x8 aumenta a resolução, mas também o ruído por bloco.
  No test, o número de acertos foi o mesmo com blocos de 8x8 e de 16x16
  (17 de 34), com duas imagens trocando de resultado.
- **Recorte ou redimensionamento depois da compressão.** Isso desalinha ou
  destrói a grade 8x8. O vale enfraquece ou some na imagem toda (não só na
  região editada).
- **q1 > q0.** Se o trecho colado tinha qualidade MAIOR que a do documento e
  depois a imagem toda foi recomprimida em q0, a compressão final apaga a
  diferença. O fantasma só sobrevive se a imagem final for salva sem perda
  (como é o caso aqui, em PNG) ou com qualidade alta.
