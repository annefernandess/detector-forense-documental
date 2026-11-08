"""
Textos da interface, em um só lugar para revisão. Regras: frases curtas, no presente, sem jargão.
Os nomes técnicos (ELA, JPEG Ghost, copy-move) só aparecem dentro dos textos de ajuda ("O que é
isto?"), sempre explicados. Nada aqui afirma que um documento foi alterado: fala-se em sinal de
atenção, possível edição e revisão.
"""

from __future__ import annotations

TITULO = "Triagem de recibos"
SUBTITULO = "Envie a imagem de um recibo. A ferramenta aponta se há sinais que merecem uma revisão humana."

ETAPAS = [
    ("Envie a imagem", "Escolha um arquivo PNG ou JPG, ou teste com um exemplo."),
    ("Veja o resultado", "A ferramenta indica se vale a pena revisar o documento."),
    ("Entenda o porquê", "Mapas sobre a imagem mostram onde estão os sinais."),
]

LIMITE_MB = 20
UPLOAD_ORIENTACAO = f"Escolha um recibo digitalizado (PNG ou JPG, até {LIMITE_MB} MB). A análise é feita na hora."
EXEMPLOS_TITULO = "Ou teste com um exemplo"
EXEMPLOS_AVISO = "Os exemplos são recibos de um conjunto de dados de pesquisa. Escolha um para ver como funciona."
ANALISANDO = "Analisando, pode levar alguns segundos..."
AGUARDANDO = "Envie uma imagem ou escolha um exemplo para começar."
DECLARACAO = ("Esta ferramenta foi validada apenas para recibos semelhantes aos do conjunto Find it again!. "
              "Ela ordena documentos para revisão e não confirma fraude.")

ERRO_FORMATO = "Este arquivo não é PNG nem JPG. Envie uma imagem em um desses formatos."
ERRO_ABRIR = "Não foi possível abrir esta imagem. Tente outro arquivo."
ERRO_ANALISE = "A análise não funcionou com esta imagem. Tente outro arquivo."


ERRO_TAMANHO = f"O arquivo é muito grande. Envie uma imagem de até {LIMITE_MB} MB."


DEMO_LINK = "modo demonstração (para apresentações)"
DEMO_AVISO = "Modo demonstração: imagens do conjunto de dados, com o gabarito em vermelho."
DEMO_SAIR = "sair do modo demonstração"
DEMO_LEGENDA = "Caixas vermelhas = gabarito do conjunto de dados. O gabarito não faz parte da análise."
DEMO_SEM_GABARITO = "O gabarito deste exemplo não marca nenhuma região editada."

PREVIA_TITULO = "Imagem analisada"
PREVIA_ALT = "Pré-visualização do recibo analisado."
PREVIA_ALT_DEMO = "Pré-visualização do exemplo, com o gabarito do conjunto de dados em vermelho."

FAIXAS = {
    "Revisar": ("Revisar", "🔎"),
    "Sem sinal relevante": ("Sem sinal relevante", "○"),
}

COMO_LER_TITULO = "Como ler o resultado"
COMO_LER_FAIXA = ("**Revisar** quer dizer que a pontuação passou do limite de atenção. "
                  "**Sem sinal relevante** quer dizer que ficou abaixo dele.")
COMO_LER_REGRA = "Esta ferramenta ordena documentos para revisão. Ela não decide se houve fraude."
POSICAO_ALT = "A comparação com os documentos de referência não está disponível nesta instalação."
POSICAO_AJUDA = ("A posição resume vários sinais da imagem; quanto mais alta, mais sinais de atenção. "
                 "Os recibos de referência são os de treino e validação do conjunto de dados, comparados por previsões "
                 "feitas sem que o próprio recibo participasse do ajuste da ferramenta.")


def posicao_frase(posicao: float) -> str:
    return f"Esta imagem está em posição mais alta que {posicao:.0f}% dos recibos de referência deste tipo."


def posicao_html(posicao: float) -> str:
    """Mesma frase de posicao_frase, com o número em destaque (classe .posicao da interface)."""
    return f"Esta imagem está em posição mais alta que <strong>{posicao:.0f}%</strong> dos recibos de referência deste tipo."


def numeros_medidos(m: dict) -> str:
    """Sensibilidade e especificidade (metricas_candidato), medidas por validação cruzada em train + val, em texto simples."""
    return (
        "Em recibos de treino e validação do conjunto de dados, cada um avaliado por um modelo que não o viu no ajuste, "
        f"a ferramenta sinalizou cerca de {m['sensibilidade']:.0%} dos documentos editados e "
        f"cerca de {1 - m['especificidade']:.0%} dos documentos sem edição. "
        "Esses números podem ser otimistas, porque os mesmos recibos ajudaram a escolher o limite de atenção."
    )


MAPAS_TITULO = "Entenda o porquê"
MAPAS_INTRO = "Cada mapa é desenhado sobre a imagem. Use o controle para ver mais ou menos do mapa."
TRANSPARENCIA = "Transparência do mapa (%)"
TRANSPARENCIA_AJUDA = "0% mostra o mapa por inteiro. 100% mostra só a imagem."
AJUDA_ROTULO = "O que é isto?"

MAPAS = {
    "ela": {
        "titulo": "Mapa de diferença de compressão",
        "legenda": "Amarelo e laranja = a imagem mudou mais ao ser comprimida de novo. Sem cor = mudou pouco.",
        "mostra": "Mostra onde a imagem reage de forma diferente do resto da página.",
        "nao_mostra": "Não prova edição: letras escuras e bordas também aparecem com cor.",
        "ajuda": ("Esta técnica se chama ELA. Ela comprime a imagem de novo e compara com a original: partes que "
                  "reagem diferente do resto podem ter sido editadas depois."),
        "alt": ("Mapa de diferença de compressão sobre o recibo. Áreas em amarelo e laranja mudaram mais ao "
                "ser comprimidas de novo; áreas sem cor mudaram pouco."),
    },
    "ghost": {
        "titulo": "Mapa de histórico de compressão",
        "legenda": "Cada cor é um nível de qualidade de compressão, do azul (mais baixo) ao oliva (mais alto).",
        "mostra": "Mostra se partes da imagem têm histórico de compressão diferente.",
        "nao_mostra": "Não diz o que foi alterado, só onde o histórico muda.",
        "ajuda": ("Esta técnica se chama JPEG Ghost. Ela procura vestígios de compressões anteriores: trechos com "
                  "histórico diferente do resto podem ter vindo de outra fonte."),
        "alt": ("Mapa de histórico de compressão sobre o recibo. Cada cor é um nível de qualidade de compressão "
                "encontrado em um bloco de texto."),
        "vazio": "Esta imagem não tem histórico JPEG, por isso este mapa não se aplica.",
        "alt_vazio": "Recibo sem mapa: esta imagem não tem histórico JPEG, por isso o mapa não se aplica.",
    },
    "copia": {
        "titulo": "Possíveis trechos repetidos",
        "legenda": "Azul e laranja = os dois trechos de um par. A linha amarela liga os dois.",
        "mostra": "Mostra trechos quase idênticos dentro da mesma imagem.",
        "nao_mostra": "Não separa cópia de texto repetido: dígitos iguais também aparecem aqui.",
        "ajuda": ("Esta técnica se chama copy-move, ou cópia e colagem. Ela compara pequenas áreas do texto entre si "
                  "e destaca as que são quase idênticas."),
        "alt": ("Possíveis trechos repetidos sobre o recibo. Retângulos azul e laranja, ligados por uma linha "
                "amarela, marcam os dois trechos de cada par."),
        "vazio": "Nenhum par de trechos repetidos foi encontrado.",
        "alt_vazio": "Recibo sem marcações: nenhum par de trechos repetidos foi encontrado.",
    },
}

NAO_FAZ_TITULO = "O que esta ferramenta não faz"
NAO_FAZ = [
    "Não confirma fraude. Ela só indica sinais para uma pessoa revisar.",
    "Não encontra edições feitas à mão ou com texto novo que não repete trechos.",
    "Funciona melhor em recibos digitalizados parecidos com os do conjunto de teste.",
]
RODAPE = "Projeto da disciplina de Processamento Digital de Imagens (UFPB)."

# Termos técnicos que só podem aparecer dentro dos textos de ajuda.
JARGAO = ("ELA", "JPEG Ghost", "copy-move", "z-score", "AUC", "IoU", "p95", "colormap")
