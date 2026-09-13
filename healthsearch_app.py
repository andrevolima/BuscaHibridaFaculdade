"""HealthSearch: BM25, embeddings locais e fusão RRF, sem Cross-Encoder."""

import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi


DOCUMENTOS = [
    (
        "Doc 1",
        "Protocolo Emergência ECG",
        "Pacientes com dor precordial aguda e suspeita de síndrome coronariana "
        "devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos.",
    ),
    (
        "Doc 2",
        "Guia de Farmacologia Cardíaca",
        "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários "
        "reduz a mortalidade no infarto agudo do miocárdio.",
    ),
    (
        "Doc 3",
        "Diretriz de Hipertensão Arterial",
        "A crise hipertensiva severa requer administração de anti-hipertensivos "
        "venosos e monitoramento contínuo da pressão arterial na UTI.",
    ),
    (
        "Doc 4",
        "Manual de AVC Isquêmico",
        "O acidente vascular cerebral isquêmico agudo deve ser tratado com "
        "trombolíticos venosos em até quatro horas e meia do início dos sintomas.",
    ),
    (
        "Doc 5",
        "Protocolo de Reanimação RCR",
        "Parada cardiorrespiratória em adultos exige compressões torácicas "
        "contínuas de alta qualidade e desfibrilação precoce no código azul.",
    ),
    (
        "Doc 6",
        "Procedimentos de UTI Geral",
        "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, "
        "recomenda-se a monitorização cardíaca contínua por telemetria.",
    ),
]

TEXTOS = [f"{titulo}. {conteudo}" for _, titulo, conteudo in DOCUMENTOS]
STOPWORDS = set(
    "a o as os e de da do das dos em no na nos nas um uma uns umas "
    "com por para ao aos ate se ser que deve devem".split()
)
MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
K_RRF = 60


def normalizar(texto):
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )


def preprocessar(texto):
    # Hífens internos são preservados: CÓD-ECG-12D vira cod-ecg-12d.
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalizar(texto))
    return [token for token in tokens if token not in STOPWORDS]


def buscar_bm25(consulta, k1, b):
    corpus = [preprocessar(texto) for texto in TEXTOS]
    bm25 = BM25Okapi(corpus, k1=k1, b=b)

    # rank_bm25 faz 0/0 para termos ausentes quando k1=0.
    # O limite correto nesse extremo é IDF por termo presente.
    if k1 == 0:
        return np.array([
            sum(
                bm25.idf.get(t, 0.0) for t in preprocessar(consulta)
                if t in documento
            )
            for documento in corpus
        ])

    return bm25.get_scores(preprocessar(consulta))


@st.cache_resource(show_spinner=False)
def carregar_modelo():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODELO, device="cpu")


@st.cache_data(show_spinner=False)
def embeddings_corpus():
    return carregar_modelo().encode(TEXTOS, convert_to_numpy=True)


def cosseno(documentos, consulta):
    denominador = np.linalg.norm(documentos, axis=1) * np.linalg.norm(consulta)
    return np.divide(
        documentos @ consulta,
        denominador,
        out=np.zeros(len(documentos)),
        where=denominador > 0,
    )


def simular_vetores(textos):
    # Alternativa didática: presença de conceitos definidos manualmente.
    grupos = [
        ("infarto", "ataque cardiaco", "sindrome coronariana", "isquemia miocardica"),
        ("acido acetilsalicilico", "aas", "aspirina", "antiagregantes"),
        ("hipertensao", "hipertensiva", "pressao alta", "pressao arterial"),
        ("avc", "derrame", "acidente vascular cerebral"),
        (
            "parada cardiorrespiratoria", "parada cardiaca", "rcr",
            "reanimacao", "codigo azul",
        ),
        ("eletrocardiograma", "ecg", "cod-ecg-12d", "ecg-12d"),
        ("arritmia", "arritmias", "telemetria", "monitorizacao", "monitoramento"),
        ("dor precordial", "dor no peito", "dor toracica"),
    ]
    return np.array([
        [
            float(any(
                re.search(r"\b" + re.escape(termo) + r"\b", normalizar(texto))
                for termo in grupo
            ))
            for grupo in grupos
        ]
        for texto in textos
    ])


@st.cache_data(show_spinner=False)
def buscar_semantico(consulta):
    try:
        modelo = carregar_modelo()
        vetor = modelo.encode(consulta, convert_to_numpy=True)
        return cosseno(embeddings_corpus(), vetor), ""
    except Exception as erro:
        # Só ativa a alternativa quando há falha real de importação/modelo.
        vetores = simular_vetores(TEXTOS + [consulta])
        return cosseno(vetores[:-1], vetores[-1]), f"{type(erro).__name__}: {erro}"


def posicoes(scores):
    # Empates seguem a ordem original (Doc 1 a Doc 6); posições começam em 1.
    ordem = np.argsort(-np.asarray(scores), kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[ordem] = np.arange(1, len(scores) + 1)
    return ranks


def fundir_rrf(bm25, semantico, alpha):
    return (
        alpha / (K_RRF + posicoes(bm25))
        + (1 - alpha) / (K_RRF + posicoes(semantico))
    )


def tabela(scores):
    return pd.DataFrame({
        "Posição": posicoes(scores),
        "Documento": [d[0] for d in DOCUMENTOS],
        "Título": [d[1] for d in DOCUMENTOS],
        "Score": scores,
        "Conteúdo": [d[2] for d in DOCUMENTOS],
    }).sort_values("Posição")


def main():
    st.set_page_config(page_title="HealthSearch", layout="wide")
    st.title("HealthSearch")
    st.caption("Busca em seis diretrizes médicas: BM25, semântico e RRF.")

    k1 = st.sidebar.slider("k1", 0.0, 3.0, 1.2, 0.1)
    b = st.sidebar.slider("b", 0.0, 1.0, 0.75, 0.05)
    alpha = st.sidebar.slider("alpha", 0.0, 1.0, 0.5, 0.05)
    st.sidebar.caption("alpha = 1: BM25; alpha = 0: semântico. k_rrf = 60.")

    consulta = st.text_input(
        "Pesquisar",
        placeholder="Ex.: CÓD-ECG-12D ou ataque cardíaco",
    ).strip()
    abas = st.tabs(["BM25", "Semântico", "Híbrido RRF", "Comparação"])

    if not preprocessar(consulta):
        st.info("Digite uma consulta com termos além de pontuação e stopwords.")
        return

    bm25 = buscar_bm25(consulta, k1, b)
    with st.spinner("Calculando busca semântica (o primeiro uso pode baixar o modelo)..."):
        semantico, erro = buscar_semantico(consulta)

    if erro:
        st.warning(
            "SIMULAÇÃO VETORIAL ATIVA: o modelo não pôde ser executado. "
            "Os vetores usam um vocabulário médico manual, não embeddings aprendidos."
        )
        with st.expander("Motivo da alternativa"):
            st.text(erro)
    else:
        st.caption(f"Modelo local: {MODELO}")

    hibrido = fundir_rrf(bm25, semantico, alpha)

    for aba, scores in zip(abas[:3], (bm25, semantico, hibrido)):
        with aba:
            st.dataframe(
                tabela(scores),
                hide_index=True,
                width="stretch",
                column_config={
                    "Score": st.column_config.NumberColumn(format="%.6f"),
                },
            )

    with abas[3]:
        comparacao = pd.DataFrame({
            "Documento": [d[0] for d in DOCUMENTOS],
            "Título": [d[1] for d in DOCUMENTOS],
        })

        for nome, scores in (("BM25", bm25), ("Semântico", semantico), ("RRF", hibrido)):
            comparacao[f"Posição {nome}"] = posicoes(scores)
            comparacao[f"Score {nome}"] = scores

        st.dataframe(
            comparacao.sort_values("Posição RRF"),
            hide_index=True,
            width="stretch",
            column_config={
                f"Score {nome}": st.column_config.NumberColumn(format="%.6f")
                for nome in ("BM25", "Semântico", "RRF")
            },
        )
        st.caption(
            "Scores têm escalas diferentes. RRF combina posições, não scores brutos. "
            "Empates seguem o ID; mesmo sem correspondência, os seis documentos são ordenados."
        )


if __name__ == "__main__":
    main()
