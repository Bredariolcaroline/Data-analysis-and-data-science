import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import numpy as np



def inspect_outliers(dataframe, column, whisker_width=1.5):
    """Função para inspecionar outliers.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    column : List[str]
        Lista com o nome das colunas (strings) a serem utilizadas.
    whisker_width : float, opcional
        Valor considerado para detecção de outliers, por padrão 1.5

    Returns
    -------
    pd.DataFrame
        Dataframe com os outliers.
    """

    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - whisker_width * iqr
    upper_bound = q3 + whisker_width * iqr

    return dataframe[
        (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
    ]





