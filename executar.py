from pickle import TRUE
import h5py
import numpy as np
import pathlib
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn import feature_selection, model_selection, neural_network
import functions
from KFoldStratifiedTargetEncoder import KFoldStratifiedTargetEncoder

PRIMEIRO_RUN = True
#PRIMEIRO_RUN_2 = True

diretorio_de_resultados = pathlib.Path().absolute() / "resultados"

if diretorio_de_resultados.exists() is False:
    diretorio_de_resultados.mkdir(exist_ok=True)

print("INICIANDO CONJUNTO DE DADOS")

if PRIMEIRO_RUN:    
    ## Leitura CSV obtidos do repositório UCI
    treino = pd.read_csv("./dados/dota2Train.csv", header=None)
    teste = pd.read_csv("./dados/dota2Test.csv", header=None)

    dados = pd.concat((treino, teste))

    X = dados.drop([0, 27, 111], axis=1).to_numpy()
    y = dados.iloc[:, 0].to_numpy()

    # Criação de um Arquigo H5 a partir de X e y
    with h5py.File("./dados/dota2.h5", "w") as arquivo:
        arquivo.create_dataset("X", data=X, compression="gzip", compression_opts=9)
        arquivo.create_dataset("y", data=y, compression="gzip", compression_opts=9)
        arquivo.create_dataset("K", data=np.unique(y).size)


nomes_das_colunas = (
        ["ID de cluster", "Modo de jogo", "Tipo de jogo"] +
        [f"Herói {i}" for i in range(26)] +
        [f"Herói {i}" for i in range(27, 111)] +
        [f"Herói {i}" for i in range(112, 113)]
)

with h5py.File("./dados/dota2.h5", "r") as arquivo:
    X = pd.DataFrame(arquivo.get("X")[()], columns=nomes_das_colunas)
    y = pd.DataFrame(arquivo.get("y")[()], columns=["Vencedor"])

print("FIM DO CONJUNTO DE DADOS")

print("INICIO DIVISAO DE TREINO E TESTE")

X_treino, X_teste, y_treino, y_teste = model_selection.train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_treino, X_teste, y_treino, y_teste = X_treino.copy(), X_teste.copy(), y_treino.copy(), y_teste.copy()

print("FIM DIVISAO DE TREINO E TESTE")

print("INICIO Conversão de variáveis em versões target-encoded")

codificador_de_rotulo = KFoldStratifiedTargetEncoder(number_of_folds=10)

X_treino.iloc[:, 0:3] = codificador_de_rotulo.fit_transform(X_treino.iloc[:, 0:3].to_numpy(),
                                                            y_treino.to_numpy().ravel())
X_teste.iloc[:, 0:3] = codificador_de_rotulo.transform(X_teste.iloc[:, 0:3].to_numpy())

print("FIM Conversão de variáveis em versões target-encoded")

print("INICIO Seleção de características")

print("PARTE 1 Seleção de características")

modelos = {
    "mlp": neural_network.MLPClassifier
}

classificadores_de_caracteristicas = {
    "ANOVA F-value": feature_selection.f_classif,
    "Mutual information": feature_selection.mutual_info_classif

}

if PRIMEIRO_RUN:
    resultados_de_classificacao = functions.classifica_caracteristicas(
        X_treino.to_numpy(), y_treino.to_numpy().ravel(), classificadores_de_caracteristicas, nomes_das_colunas)

    resultados_de_classificacao.to_csv(
        f"{diretorio_de_resultados.as_posix()}/Classificações de características.csv", sep=";")


resultados_de_classificacao = pd.read_csv(
    f"{diretorio_de_resultados.as_posix()}/Classificações de características.csv", sep=";", index_col=0
)

figura, eixos = plt.subplots(1, resultados_de_classificacao.shape[1], figsize=(25, 25), squeeze=True)

for indice, coluna in enumerate(resultados_de_classificacao.columns):
    resultados_de_classificacao.loc[:, coluna].sort_values().plot.barh(
        color="steelblue", title=coluna, ax=eixos.ravel()[indice]
    )

figura.savefig(f"{diretorio_de_resultados.as_posix()}/Classificações de características.jpg", transparent=False)

print("PARTE 2 Seleção de características")

particionador = model_selection.StratifiedKFold(n_splits=10)
tamanhos = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 114]

if PRIMEIRO_RUN:
    resultados_de_validacao = functions.testa_conjuntos_de_caracteristicas(
        X_treino.to_numpy(), y_treino.to_numpy().ravel(), modelos, tamanhos, resultados_de_classificacao, particionador, verboso=True)
    resultados_de_validacao.to_csv(
        f"{diretorio_de_resultados.as_posix()}/Medidas de modelos.csv", sep=";", index=False)

resultados_de_validacao = pd.read_csv(
    f"{diretorio_de_resultados.as_posix()}/Medidas de modelos.csv", sep=";", encoding="utf-8"
)
resultados_de_validacao.sort_values(by="Pontuacao final", ascending=False).head(10)

print("Range da acurácia média de teste:     "
      f"[{resultados_de_validacao.loc[:, 'Acuracia media de teste'].min().round(4)}, "
      f"{resultados_de_validacao.loc[:, 'Acuracia media de teste'].max().round(4)}]\n"
      "Range da pontuação F1 média de teste: "
      f"[{resultados_de_validacao.loc[:, 'Pontuacao F1 media de teste'].min().round(4)}, "
      f"{resultados_de_validacao.loc[:, 'Pontuacao F1 media de teste'].max().round(4)}]\n"
      "Range da pontuação final:             "
      f"[{resultados_de_validacao.loc[:, 'Pontuacao final'].min().round(4)}, "
      f"{resultados_de_validacao.loc[:, 'Pontuacao final'].max().round(4)}]\n")

figura, eixos = plt.subplots(1, 3, figsize=(25, 6))

sns.lineplot(data=resultados_de_validacao, x="Tamanho", y="Acuracia media de teste",
             hue="Classificador de caracteristicas", ax=eixos[0])
sns.lineplot(data=resultados_de_validacao, x="Tamanho", y="Pontuacao F1 media de teste",
             hue="Classificador de caracteristicas", ax=eixos[1])
sns.lineplot(data=resultados_de_validacao, x="Tamanho", y="Pontuacao final", hue="Classificador de caracteristicas",
             ax=eixos[2])

figura.savefig(f"{diretorio_de_resultados.as_posix()}/Informação dos modelos (por classificador).jpg",
               transparent=False)

figura, eixos = plt.subplots(1, 3, figsize=(25, 6))

sns.lineplot(data=resultados_de_validacao, x="Tamanho", y="Acuracia media de teste", hue="Algoritmo", ax=eixos[0])
sns.lineplot(data=resultados_de_validacao, x="Tamanho", y="Pontuacao F1 media de teste", hue="Algoritmo", ax=eixos[1])
sns.lineplot(data=resultados_de_validacao, x="Tamanho", y="Pontuacao final", hue="Algoritmo", ax=eixos[2])

figura.savefig(f"{diretorio_de_resultados.as_posix()}/Informação dos modelos (por algoritmo).jpg", transparent=False)

figura, eixo = plt.subplots(1, 1, figsize=(12, 6))

sns.lineplot(data=resultados_de_validacao, x="Tamanho", y="Tempo medio de treino", hue="Algoritmo", ax=eixo)

figura.savefig(f"{diretorio_de_resultados.as_posix()}/Tendência do tempo de treinamento.jpg", transparent=False)

pontuacoes = resultados_de_classificacao.loc[:, "ANOVA F-value"].to_numpy()
indices_organizados = pontuacoes.argsort()[::-1]

contagem_de_rotulos_treino = y_treino.value_counts()
print(contagem_de_rotulos_treino)

contagem_de_rotulos_teste = y_teste.value_counts()
print(contagem_de_rotulos_teste)

print(contagem_de_rotulos_treino.iloc[0] / contagem_de_rotulos_treino.sum())

print(contagem_de_rotulos_teste.iloc[0] / contagem_de_rotulos_teste.sum())

print("FIM SELECAO DE CARACTERISTICAS")

print("INICIO MODELOS CONSTRUIDOS")

if PRIMEIRO_RUN:
    resultados_de_teste = functions.gera_resultados(X, y, modelos, {"mlp": 30}, resultados_de_classificacao, "ANOVA F-value")
    resultados_de_teste.to_csv(
        f"{diretorio_de_resultados.as_posix()}/Resultados de teste.csv", sep=";")

resultados_de_teste = pd.read_csv(f"{diretorio_de_resultados.as_posix()}/Resultados de teste.csv", sep=";", index_col=[0, 1], header=[0, 1])

print(resultados_de_teste)

acuracia_media = functions.obtem_medida_media(resultados_de_teste, "Acurácia", modelos)
print(acuracia_media)

print("FIM MODELOS CONSTRUIDOS")
