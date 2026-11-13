"""Classificação de países por prioridade de ajuda — UNIFSA, 2026.2."""
from datetime import date
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Classificação de países", layout="centered")

PASTA = Path(__file__).resolve().parent


@st.cache_resource
def carregar_modelo():
    return joblib.load(PASTA / "modelo_paises.pkl")


pacote = carregar_modelo()
modelo = pacote["modelo"]
colunas = pacote["colunas"]
nomes = pacote["nomes"]

CAMPOS = {
    "child_mort": ("Mortalidade infantil (mortes por 1.000 nascidos vivos)", 19.3),
    "exports": ("Exportações (% do PIB per capita)", 35.0),
    "health": ("Gasto com saúde (% do PIB per capita)", 6.3),
    "imports": ("Importações (% do PIB per capita)", 43.3),
    "income": ("Renda líquida por pessoa (US$)", 9960.0),
    "inflation": ("Inflação anual (%)", 5.4),
    "life_expec": ("Expectativa de vida (anos)", 73.1),
    "total_fer": ("Filhos por mulher", 2.4),
    "gdpp": ("PIB per capita (US$)", 4660.0),
}

CORES = {
    "Prioridade alta": "#BA4646",
    "Prioridade intermediária": "#9A6719",
    "Prioridade baixa": "#246FA8",
}

st.title("Classificação de países")
st.write("Informe os indicadores do país para obter a prioridade estimada pelo modelo.")

with st.form("classificacao"):
    pais = st.text_input("Nome do país (opcional)")
    valores = {}
    esquerda, direita = st.columns(2)
    for i, coluna in enumerate(colunas):
        rotulo, inicial = CAMPOS[coluna]
        with esquerda if i % 2 == 0 else direita:
            valores[coluna] = st.number_input(rotulo, value=inicial, key=coluna, format="%.3f")
    enviado = st.form_submit_button("Classificar", type="primary")

if enviado:
    entrada = pd.DataFrame([valores], columns=colunas)
    if not np.isfinite(entrada.to_numpy()).all():
        st.error("Informe valores numéricos finitos em todos os indicadores.")
    else:
        pais = pais.strip() or "País informado"
        grupo = int(modelo.predict(entrada)[0])
        st.success(f"Grupo {grupo}: {nomes[grupo]}")

        st.subheader("Comparação com o grupo atribuído")
        st.caption("As médias foram calculadas sobre os países do grupo na base de treino, nas unidades originais.")
        comparacao = pd.DataFrame({
            "Indicador": [CAMPOS[c][0] for c in colunas],
            "País informado": entrada.iloc[0].to_numpy(),
            "Média do grupo": pacote["medias"].loc[grupo, colunas].to_numpy(),
        })
        st.dataframe(comparacao, hide_index=True, column_config={
            "País informado": st.column_config.NumberColumn(format="%.2f"),
            "Média do grupo": st.column_config.NumberColumn(format="%.2f"),
        })

        st.subheader("Posição no plano PC1 × PC2")
        coords = modelo[:-1].transform(entrada)[0]
        pc1, pc2 = coords[:2]
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for g, pontos in pacote["referencia"].groupby("Grupo"):
            ax.scatter(pontos["PC1"], pontos["PC2"], color=CORES[nomes[g]],
                       s=26, alpha=.65, label=nomes[g])
        ax.scatter(pc1, pc2, marker="*", s=235, color="#171717", edgecolors="white",
                   linewidths=1, label="País informado", zorder=5)
        ax.set(xlabel="PC1", ylabel="PC2")
        ax.grid(alpha=.2)
        ax.legend(loc="best")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"{pais}: PC1 = {pc1:.3f} | PC2 = {pc2:.3f}. "
                   "A classificação usa os cinco componentes; o gráfico mostra apenas os dois primeiros.")

st.header("Sobre o modelo")
st.markdown(f"**Origem dos dados:** [Unsupervised Learning on Country Data — Kaggle]({pacote['fonte']}).")
st.write(f"**Data de treinamento:** {date.fromisoformat(pacote['data_treinamento']):%d/%m/%Y}")
st.write(f"**Componentes do PCA:** {modelo.named_steps['pca'].n_components_}")
st.write(f"**Número de grupos:** {modelo.named_steps['kmeans'].n_clusters}")
st.caption("Modelo exploratório de apoio à análise. A prioridade estimada não substitui a avaliação das necessidades locais.")
