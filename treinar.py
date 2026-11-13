"""Treina o pipeline e gera todos os dados usados pelo aplicativo."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

PASTA = Path(__file__).resolve().parent
FONTE = "https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data"


def treinar():
    base = pd.read_csv(PASTA / "country-data.csv")
    X = base.drop(columns="country")
    modelo = make_pipeline(
        StandardScaler(),
        PCA(n_components=5, svd_solver="full"),
        KMeans(n_clusters=3, n_init=10, random_state=42),
    )
    with threadpool_limits(limits=1):
        modelo.fit(X)
        grupos = modelo.predict(X)
    data = datetime.now(ZoneInfo("America/Fortaleza")).date().isoformat()
    medias = X.groupby(grupos).mean()
    ordem = medias["child_mort"].sort_values(ascending=False).index
    nomes = dict(zip(map(int, ordem), [
        "Prioridade alta", "Prioridade intermediária", "Prioridade baixa",
    ]))
    proj = modelo[:-1].transform(X)
    referencia = pd.DataFrame({
        "País": base["country"], "PC1": proj[:, 0], "PC2": proj[:, 1], "Grupo": grupos,
    })
    pacote = {
        "modelo": modelo, "colunas": list(X.columns), "nomes": nomes,
        "medias": medias, "referencia": referencia,
        "fonte": FONTE, "data_treinamento": data,
    }
    joblib.dump(pacote, PASTA / "modelo_paises.pkl")
    return pacote


if __name__ == "__main__":
    treinar()
    print("Arquivo modelo_paises.pkl gerado.")
