import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi


DOCUMENTOS = [
    (
        "Doc 1",
        "Protocolo Emergencia ECG",
        "Pacientes com dor precordial aguda e suspeita de sindrome coronariana "
        "devem realizar eletrocardiograma COD-ECG-12D em ate 10 minutos.",
    ),
    (
        "Doc 2",
        "Guia de Farmacologia Cardiaca",
        "O uso imediato de acido acetilsalicilico e antiagregantes plaquetarios "
        "reduz a mortalidade no infarto agudo do miocardio.",
    ),
    (
        "Doc 3",
        "Diretriz de Hipertensao Arterial",
        "A crise hipertensiva severa requer administracao de anti-hipertensivos "
        "venosos e monitoramento continuo da pressao arterial na UTI.",
    ),
    (
        "Doc 4",
        "Manual de AVC Isquemico",
        "O acidente vascular cerebral isquemico agudo deve ser tratado com "
        "tromboliticos venosos em ate quatro horas e meia do inicio dos sintomas.",
    ),
    (
        "Doc 5",
        "Protocolo de Reanimacao RCR",
        "Parada cardiorrespiratoria em adultos exige compressoes toracicas "
        "continuas de alta qualidade e desfibrilacao precoce no codigo azul.",
    ),
    (
        "Doc 6",
        "Procedimentos de UTI Geral",
        "Para diagnostico do protocolo COD-ECG-12D em arritmias complexas, "
        "recomenda-se a monitorizacao cardiaca continua por telemetria.",
    ),
]

TEXTOS = [f"{titulo}. {conteudo}" for _, titulo, conteudo in DOCUMENTOS]
STOPWORDS = set(
    "a o as os e de da do das dos em no na nos nas um uma uns umas "
    "com por para ao aos ate se ser que deve devem".split()
)
K_RRF = 60


def normalizar(texto):
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )


def preprocessar(texto):
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalizar(texto))
    return [token for token in tokens if token not in STOPWORDS]


def buscar_bm25(consulta, k1, b):
    corpus = [preprocessar(texto) for texto in TEXTOS]
    bm25 = BM25Okapi(corpus, k1=k1, b=b)

    if k1 == 0:
        return np.array([
            sum(
                bm25.idf.get(t, 0.0) for t in preprocessar(consulta)
                if t in documento
            )
            for documento in corpus
        ])

    return bm25.get_scores(preprocessar(consulta))


def cosseno(documentos, consulta):
    denominador = np.linalg.norm(documentos, axis=1) * np.linalg.norm(consulta)
    return np.divide(
        documentos @ consulta,
        denominador,
        out=np.zeros(len(documentos)),
        where=denominador > 0,
    )


def simular_vetores(textos):
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
    vetores = []
    for texto in textos:
        texto = normalizar(texto)
        vetor = []
        for grupo in grupos:
            encontrou = any(
                re.search(r"\b" + re.escape(termo) + r"\b", texto)
                for termo in grupo
            )
            vetor.append(float(encontrou))
        vetores.append(vetor)
    return np.array(vetores)


def posicoes(scores):
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
        "Posicao": posicoes(scores),
        "Documento": [d[0] for d in DOCUMENTOS],
        "Titulo": [d[1] for d in DOCUMENTOS],
        "Score": scores,
        "Conteudo": [d[2] for d in DOCUMENTOS],
    }).sort_values("Posicao")


def main():
    st.title("HealthSearch")
    st.write(
        "Busca semantica por simulacao vetorial, permitida no enunciado. "
        "Usa conceitos e sinonimos manuais, nao um modelo treinado."
    )

    k1 = st.sidebar.slider("k1", 0.0, 3.0, 1.2, 0.1)
    b = st.sidebar.slider("b", 0.0, 1.0, 0.75, 0.05)
    alpha = st.sidebar.slider("alpha", 0.0, 1.0, 0.5, 0.05)
    st.sidebar.caption("alpha = 1: BM25; alpha = 0: semantico. k_rrf = 60.")

    consulta = st.text_input("Pesquisar", placeholder="COD-ECG-12D ou derrame")
    abas = st.tabs(["BM25", "Semantico", "Hibrido RRF", "Comparacao"])
    if not preprocessar(consulta):
        return

    bm25 = buscar_bm25(consulta, k1, b)
    vetores = simular_vetores(TEXTOS + [consulta])
    semantico = cosseno(vetores[:-1], vetores[-1])
    hibrido = fundir_rrf(bm25, semantico, alpha)

    resultados = [("BM25", bm25), ("Semantico", semantico), ("RRF", hibrido)]
    comparacao = pd.DataFrame({
        "Documento": [doc[0] for doc in DOCUMENTOS],
        "Titulo": [doc[1] for doc in DOCUMENTOS],
    })

    for aba, (nome, scores) in zip(abas[:3], resultados):
        with aba:
            st.table(tabela(scores).reset_index(drop=True))
        comparacao[f"Posicao {nome}"] = posicoes(scores)
        comparacao[f"Score {nome}"] = scores

    with abas[3]:
        st.table(comparacao.sort_values("Posicao RRF").reset_index(drop=True))
        st.write("RRF combina posicoes. Empates seguem o ID do documento.")
        st.write("Todos os documentos sao listados; scores nao sao probabilidades.")


if __name__ == "__main__":
    main()
