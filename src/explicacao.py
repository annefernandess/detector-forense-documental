"""
Explicação em texto, por regras, dos resultados de ELA (regiões), copy-move (pares) e JPEG Ghost
(histórico JPEG). Uma ou duas frases, sempre em linguagem de indicação ("pode indicar edição",
"possível repetição"), sem afirmar que o documento foi alterado. A faixa do classificador não entra
no texto como afirmação: só evita uma contradição quando ela pede revisão e nenhuma região foi
destacada.
"""

from __future__ import annotations

# Limiares fixos, escolhidos só no val (app/calibrar_regioes.py, results/regioes_config.json):
# a partir de 5 pares, a imagem tem mais pares do que o percentil 99 (4,3) das autênticas do val.
MUITOS_PARES = 5

_POSICOES = [["no canto superior esquerdo", "na parte superior", "no canto superior direito"],
             ["no lado esquerdo", "no centro", "no lado direito"],
             ["no canto inferior esquerdo", "na parte inferior", "no canto inferior direito"]]


def posicao(caixa: tuple, tamanho: tuple) -> str:
    """Posição da caixa (x, y, w, h) na imagem de tamanho (largura, altura), em nove zonas."""
    x, y, w, h = caixa
    linha = min(2, int(3 * (y + h / 2) / tamanho[1]))
    coluna = min(2, int(3 * (x + w / 2) / tamanho[0]))
    return _POSICOES[linha][coluna]


def _maiuscula(texto: str) -> str:
    return texto[0].upper() + texto[1:]


def explicar(regioes: list, tamanho: tuple, pares: list, ghost_historico: bool, faixa: str) -> str:
    """
    Frases sobre as regiões de ELA (`regioes`: lista de ((x, y, w, h), intensidade), a primeira é a
    de maior intensidade), os pares de copy-move (`pares`), o histórico JPEG e a faixa do
    classificador ("Revisar" ou "Sem sinal relevante"). `tamanho` é (largura, altura) da imagem.
    """
    if not regioes and not pares:
        primeira = "Nenhuma região com diferença de compressão acima do esperado foi encontrada."
    else:
        if not regioes:
            ela = "nenhuma região com diferença de compressão acima do esperado foi encontrada"
        elif len(regioes) == 1:
            ela = (f"há uma região de maior diferença de compressão {posicao(regioes[0][0], tamanho)}, "
                   "o que pode indicar edição")
        else:
            ela = (f"há {len(regioes)} regiões de maior diferença de compressão, a de maior intensidade "
                   f"{posicao(regioes[0][0], tamanho)}, o que pode indicar edição")
        n = len(pares)
        if n == 0:
            repeticao = "não foi encontrada repetição de trecho"
        elif n == 1:
            repeticao = "foi encontrado um possível trecho repetido entre duas regiões, cuja origem deve ser conferida visualmente"
        elif n < MUITOS_PARES:
            repeticao = f"foram encontrados {n} possíveis trechos repetidos, que devem ser conferidos visualmente"
        else:
            repeticao = (f"foram encontrados {n} pares de trechos quase idênticos, comuns em texto repetitivo "
                         "(dígitos, tabelas), por isso a conferência visual é necessária")
        primeira = f"{_maiuscula(ela)}; {repeticao}."

    notas = []
    if not ghost_historico:
        notas.append("o JPEG Ghost não se aplica a esta imagem (sem histórico JPEG)")
    if faixa == "Revisar" and not regioes and not pares:
        notas.append("a pontuação do classificador indica revisão, mas nenhuma região específica foi destacada, "
                     "então a conferência deve olhar o documento como um todo")
    return primeira if not notas else f"{primeira} {_maiuscula('; '.join(notas))}."


def resumo(faixa: str, n_pares: int) -> str:
    """
    Frase de resumo, em linguagem simples, para a tela de resultado. Fala em sinais de atenção e
    em revisão; não usa nomes de técnicas e não afirma que o documento foi alterado. `faixa` é
    "Revisar" ou "Sem sinal relevante"; `n_pares` é o número de possíveis trechos repetidos.
    """
    if faixa == "Revisar":
        base = "Esta imagem tem sinais de atenção que merecem uma revisão humana."
        return base + (" Há também possíveis trechos repetidos." if n_pares else "")
    if n_pares:
        return ("Esta imagem não passou do limite de atenção, mas há possíveis trechos repetidos para conferir. "
                "Isso não garante que o documento esteja intacto.")
    return "Esta imagem não tem sinais fortes de atenção. Isso não garante que o documento esteja intacto."
