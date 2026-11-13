# Classificação de países — atividades complementares

Aplicativo Streamlit para estimar o grupo de prioridade de um país a partir de nove indicadores.

## Executar

Use Python 3.11 ou superior. Extraia o ZIP, abra um terminal na pasta `projeto-ia` e rode:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

O modelo e a base já acompanham o projeto. Não é necessário baixar o CSV nem treinar o modelo para executar o aplicativo. Para treinar novamente, use `python treinar.py`.

## Funcionalidades

1. Exibe o grupo atribuído e compara os nove indicadores do país com as médias dos países de referência do grupo, nas unidades originais.
2. Projeta o país com `modelo[:-1].transform(entrada)` e destaca sua posição em PC1 × PC2. A classificação usa os cinco componentes.
3. Exibe a seção **Sobre o modelo** com a fonte, a data de treinamento, os cinco componentes e os três grupos.

A base contém 167 países e vem do conjunto [Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data), publicado no Kaggle. A classificação é exploratória e não substitui a avaliação de necessidades locais.
