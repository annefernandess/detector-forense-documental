# Fundamentação teórica — Copy-move por correlação normalizada

Ideia: na adulteração do tipo CPI, um trecho do próprio documento é copiado e
colado em outro ponto. A região suspeita é usada como máscara e comparada com
cada posição da imagem por correlação normalizada. Se ela foi copiada de outro
ponto do documento, a correlação com esse ponto fica próxima de 1, e a posição
do máximo indica a origem provável da cópia.

Referências de partida: material da disciplina (slides 260-272, correlação e
filtro casado) e Fridrich, Soukal e Lukáš (2003), que propõem a detecção de
copy-move pela comparação de blocos da imagem.


## Fundamentação teórica

### 1. Coeficiente de correlação de Pearson

O coeficiente de correlação linear de Pearson entre dois vetores a e b mede a
semelhança entre eles depois de retirar a média e normalizar a escala
(slide 260):

```
r = (a − μa) · (b − μb) / ( ‖a − μa‖ · ‖b − μb‖ )
```

O valor de r fica entre −1 e 1. Ele vale 1 quando b é uma versão de a com
ganho positivo e deslocamento quaisquer (b = α·a + β, com α > 0). O slide 270
mostra esse efeito: a máscara (3, 7, 5) tem correlação 1 com (3, 7, 5) e também
com (6, 14, 10). Para imagens, isso significa que a medida não depende de
diferenças globais de brilho e contraste entre os dois trechos.

### 2. Correlação normalizada e filtro casado

A correlação normalizada entre um sinal s e uma máscara h é um sinal g em que
cada amostra é o coeficiente de Pearson entre h e o trecho de s sob a máscara,
naquela posição (slide 266). Em duas dimensões, para uma máscara T de tamanho
w x h e a janela W(u, v) da imagem com canto superior esquerdo em (u, v):

```
g(u, v) = Σ [W(u, v) − μW] · [T − μT] / sqrt( Σ [W(u, v) − μW]² · Σ [T − μT]² )
```

O **filtro casado** usa essa operação para localizar um padrão: a resposta é
máxima na posição do sinal mais semelhante à máscara (slide 270).

Quando a janela é constante (por exemplo, papel em branco), o denominador é
zero e a correlação é indefinida; o slide 270 mostra esse caso como NaN. Na
implementação, essas posições recebem −1, para nunca serem escolhidas como
máximo.

A correlação **não** normalizada (produto interno, slide 273) não serve para
este fim: ela cresce com o brilho e o contraste da janela (slide 284), e o
máximo tenderia a cair nas regiões mais escuras, e não nas mais semelhantes.

### 3. Aplicação ao copy-move

1. A região suspeita é a máscara T.
2. Calcula-se g(u, v) para todas as posições da imagem.
3. As posições em que a janela sobrepõe a própria região são excluídas, porque
   a correlação de uma região com ela mesma é sempre 1.
4. O máximo de g é o **pico**, e a posição do máximo é a origem provável.

**Por que uma cópia dá r ≈ 1 e o texto repetido não.** Copiar e colar em um
editor de imagem duplica os pixels, incluindo o ruído da digitalização. Um
caractere impresso várias vezes passa por impressão e digitalização separadas,
com ruído, deslocamento de fração de pixel e variações de tinta diferentes, e a
correlação fica abaixo de 1. No split `val`, a mediana do pico é 0,997 nas
regiões CPI e 0,855 em regiões de controle com texto (notebook
`03_copy_move.ipynb`).

**Decisão.** Uma região é considerada cópia quando o pico é maior ou igual a um
limiar. O limiar é escolhido no `val` pelo índice de Youden (J = TPR − FPR),
em que TPR é a fração de regiões CPI acima do limiar e FPR a fração de
controles acima do limiar, e depois aplicado uma única vez no `test`.

**Implementação.** A função `cv2.matchTemplate` com o modo
`TM_CCOEFF_NORMED` calcula exatamente g(u, v): subtrai as médias da janela e da
máscara e normaliza pelo produto das normas.

### 4. Limitações

- **Caixas maiores que o trecho colado.** Se a caixa anotada inclui fundo ou
  parte de caracteres vizinhos, essa parte não se repete na origem, e a
  correlação cai. No notebook, o mesmo dígito copiado tem r = 1,000 com uma
  caixa justa e r = 0,828 com uma caixa um pouco mais larga.
- **Cópias transformadas.** A correlação normalizada é invariante a brilho e
  contraste, mas não a escala, rotação ou retoque. Uma cópia redimensionada ou
  retocada depois de colada (CPI combinado com PIX) tem correlação menor.
- **Regiões lisas.** Uma edição que apenas apaga conteúdo (CUT) deixa uma região
  sem textura, em que a correlação é indefinida.
- **Cópias múltiplas.** Quando o mesmo trecho é colado em vários lugares, o pico
  pode cair em outra cópia, e não na origem.
- **Repetições exatas que não vêm de cópia.** Texto inserido com a ferramenta de
  texto de um editor (IMI) pode produzir caracteres idênticos pixel a pixel. No
  `test`, as duas imagens sem CPI detectadas têm apenas adulterações do tipo IMI.
- **Cópia de outro documento (CPO).** A origem não está na imagem, e a técnica
  não tem com o que comparar.
- **Avaliação com as caixas anotadas.** A avaliação mede se a região modificada
  tem uma cópia quase exata, e não se o método encontra a região sem as
  anotações. Uma busca cega exige comparar todos os trechos da imagem entre si,
  como na comparação de blocos de Fridrich et al. (2003), e não foi feita.

### 5. Relação com o material da disciplina

Páginas de `material/PDI2026.2.pdf`, conferidas por busca no texto do PDF:

- **Coeficiente de correlação de Pearson:** slide 260.
- **Correlação normalizada:** definição no slide 266 e exercício nos slides
  267-270, incluindo o caso indefinido (NaN) e a invariância a ganho.
- **Filtro casado:** slides 270-272.
- **Correlação não normalizada:** slides 273-284.
- **Fora do material:** o índice de Youden não aparece nos slides e precisa de
  outra fonte no relatório (Youden, 1950).
