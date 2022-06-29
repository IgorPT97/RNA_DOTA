import numpy as np
import pandas as pd
from sklearn import feature_selection, metrics, model_selection
from KFoldStratifiedTargetEncoder import KFoldStratifiedTargetEncoder

def classifica_caracteristicas(X, y, classificadores, nomes_das_caracteristicas=None, repeticoes=100):
    resultados = np.empty((X.shape[1], len(classificadores)))
    nomes_das_colunas = classificadores.keys()

    if nomes_das_caracteristicas is None:
        nomes_das_linhas = [i for i in range(X.shape[1])]
    else:
        nomes_das_linhas = nomes_das_caracteristicas

    for indice, itens in enumerate(classificadores.items()):
        nome, classificador = itens

        if nome == "Mutual information":
            pontuacoes = np.empty((repeticoes, X.shape[1]))

            for repeticao in range(repeticoes):
                print(f"Repetição {repeticao+1}/{repeticoes} - Classificador Mutual Information")
                seletor_dos_k_melhores = feature_selection.SelectKBest(classificador, k=114)
                seletor_dos_k_melhores.fit(X, y)

                pontuacoes[repeticao, :] = seletor_dos_k_melhores.scores_

            resultados[:, indice] = pontuacoes.mean(0)

            # print(f"Desvio padrão médio: {pontuacoes.std(0).mean()}")
        else:
            seletor_dos_k_melhores = feature_selection.SelectKBest(classificador, k=114)
            seletor_dos_k_melhores.fit(X, y)

            resultados[:, indice] = seletor_dos_k_melhores.scores_

    resultados = pd.DataFrame(resultados, index=nomes_das_linhas, columns=nomes_das_colunas)

    return resultados


def testa_conjuntos_de_caracteristicas(X, y, modelos, tamanhos, pontuacoes_de_classificacao, particionador,
                                       verboso=False):
    resultados = np.empty((0, 7), dtype=np.object_)

    for classificador in pontuacoes_de_classificacao.columns:
        pontuacoes = pontuacoes_de_classificacao.loc[:, classificador].to_numpy()
        indices_organizados = pontuacoes.argsort()[::-1]

        for tamanho in tamanhos:
            novo_X = X[:, indices_organizados[:tamanho]]

            for acronimo, modelo in modelos.items():
                if verboso is True:
                    print(f"[Tamanho = {tamanho}] Executando {acronimo.upper()}..." + 5 * " ", end="\r")

                if acronimo == "mlp":
                    instancia_do_modelo = modelo(hidden_layer_sizes=((novo_X.shape[1] + 1) // 2,), random_state=42)
                elif acronimo == "dt":
                    instancia_do_modelo = modelo()

                validacao_cruzada = model_selection.cross_validate(
                    instancia_do_modelo, novo_X, y, n_jobs=-1, cv=particionador, scoring=[
                        "accuracy", "f1"
                    ]
                )

                resultado = np.array([[
                    tamanho,
                    acronimo.upper(),
                    classificador,
                    validacao_cruzada["fit_time"].mean(),
                    validacao_cruzada["test_accuracy"].mean(),
                    validacao_cruzada["test_f1"].mean(),
                    np.array([
                        validacao_cruzada["test_accuracy"].mean(),
                        validacao_cruzada["test_f1"].mean()
                    ]).mean()
                ]])
                resultados = np.concatenate((resultados, resultado), axis=0)

    if verboso is True:
        print("Execução concluída." + 20 * " ")

    nomes_das_colunas = [
        "Tamanho",
        "Algoritmo",
        "Classificador de caracteristicas",
        "Tempo medio de treino",
        "Acuracia media de teste",
        "Pontuacao F1 media de teste",
        "Pontuacao final"
    ]

    resultados = pd.DataFrame(resultados, columns=nomes_das_colunas)

    for coluna in nomes_das_colunas:
        if coluna not in nomes_das_colunas[1:3]:
            resultados.loc[:, coluna] = pd.to_numeric(resultados.loc[:, coluna])

    return resultados

def gera_resultados(X, y, modelos, tamanho, pontuacoes_de_classificacao, classificador, verboso=False, execucoes=10):
    bases = ["Base de treino", "Base de teste", "Base completa"]
    iteracoes = list(range(execucoes))

    algoritmos = [acronimo.upper() for acronimo in modelos.keys()]
    nomes_das_medidas = [
        "Matriz de confusão", "Sensibilidade", "Especificidade", "Confiabilidade positiva", "Confiabilidade negativa",
        "Acurácia"
    ]

    nomes_das_linhas = pd.MultiIndex.from_product((bases, iteracoes), names=["Base", "Execução"])
    nomes_das_colunas = pd.MultiIndex.from_product((algoritmos, nomes_das_medidas), names=["Algoritmo", "Medida"])

    resultados = pd.DataFrame(index=nomes_das_linhas, columns=nomes_das_colunas)

    for i in range(execucoes):
        for acronimo in modelos.keys():
            dados = gera_dados(X, y, tamanho[acronimo], pontuacoes_de_classificacao, classificador, semente=i)
            X_filtrado, y_filtrado, X_treino, X_teste, y_treino, y_teste = dados

            modelo_treinado = \
            treina_modelo(X_treino, y_treino, {acronimo: modelos[acronimo]}, verboso=verboso, semente=i)[acronimo]

            dados = {"Base de treino": X_treino, "Base de teste": X_teste, "Base completa": X_filtrado}
            rotulos = {"Base de treino": y_treino, "Base de teste": y_teste, "Base completa": y_filtrado}
            for base in bases:
                y_pred = modelo_treinado.predict(dados[base])

                matriz_de_confusao = metrics.confusion_matrix(rotulos[base], y_pred)
                verdadeiro_negativo, falso_positivo, falso_negativo, verdadeiro_positivo = matriz_de_confusao.ravel()

                sensibilidade = verdadeiro_positivo / (verdadeiro_positivo + falso_negativo)
                especificidade = verdadeiro_negativo / (verdadeiro_negativo + falso_positivo)
                confiabilidade_positiva = verdadeiro_positivo / (verdadeiro_positivo + falso_positivo)
                confiabilidade_negativa = verdadeiro_negativo / (verdadeiro_negativo + falso_negativo)
                acuracia = metrics.accuracy_score(rotulos[base], y_pred)

                medidas = np.array(
                    [sensibilidade, especificidade, confiabilidade_positiva, confiabilidade_negativa, acuracia]
                )

                resultados.loc[
                    (base, i), (acronimo.upper(), "Matriz de confusão")
                ] = str(matriz_de_confusao).replace("\n", ",").replace("[ ", "[")
                resultados.loc[
                    (base, i), (acronimo.upper(), nomes_das_medidas[1:])
                ] = medidas

            for nome_da_medida in nomes_das_medidas[1:]:
                resultados.loc[:, (acronimo.upper(), nome_da_medida)] = pd.to_numeric(
                    resultados.loc[:, (acronimo.upper(), nome_da_medida)]
                )

    return resultados

def gera_dados(X, y, tamanho, pontuacoes_de_classificacao, classificador, semente=None):
    pontuacoes = pontuacoes_de_classificacao.loc[:, classificador].to_numpy()
    indices_organizados = pontuacoes.argsort()[::-1]

    X_filtrado = X.iloc[:, indices_organizados[:tamanho]].copy()

    X_treino, X_teste, y_treino, y_teste = model_selection.train_test_split(
        X_filtrado, y, test_size=0.3, random_state=semente, stratify=y
    )
    X_treino, X_teste, y_treino, y_teste = X_treino.copy(), X_teste.copy(), y_treino.copy(), y_teste.copy()

    codificador_de_rotulo = KFoldStratifiedTargetEncoder(number_of_folds=10)

    X_treino.iloc[:, 0:3] = codificador_de_rotulo.fit_transform(
        X_treino.iloc[:, 0:3].to_numpy(), y_treino.to_numpy().ravel()
    )
    X_teste.iloc[:, 0:3] = codificador_de_rotulo.transform(X_teste.iloc[:, 0:3].to_numpy())

    X_treino, y_treino = X_treino.to_numpy(), y_treino.to_numpy().ravel()
    X_teste, y_teste = X_teste.to_numpy(), y_teste.to_numpy().ravel()

    return X_filtrado.to_numpy(), y.to_numpy().ravel(), X_treino, X_teste, y_treino, y_teste


def treina_modelo(X, y, modelos, verboso=False, semente=None):
    instancias_de_modelos = {}
    for acronimo, classe in modelos.items():
        if verboso is True:
            print(f"Executando {acronimo.upper()}..." + 5 * " ", end="\r")

        if acronimo == "mlp":
            modelo = classe(hidden_layer_sizes=((X.shape[1] + 1) // 2,), random_state=semente)


        modelo.fit(X, y)
        instancias_de_modelos[acronimo] = modelo

    if verboso is True:
        print("Execução concluída." + 20 * " ")

    return instancias_de_modelos


def obtem_medida_media(resultados_de_teste, medida, modelos):
    nomes_das_linhas = ["Base de treino", "Base de teste", "Base completa"]
    nomes_das_colunas = [nome.upper() for nome in modelos.keys()]

    media = resultados_de_teste.xs(medida, axis=1, level=1).groupby("Base").mean().to_numpy()
    desvio_padrao = resultados_de_teste.xs(medida, axis=1, level=1).groupby("Base").std().to_numpy()

    medida_media = pd.DataFrame(
        index=nomes_das_linhas, columns=nomes_das_colunas, dtype=np.object_
    )

    medida_media.index.name = "Base"
    medida_media.columns.name = "Algoritmos"

    for l in range(len(nomes_das_linhas)):
        for c in range(len(nomes_das_colunas)):
            str_media = str((media[l, c] * 100).round(3)).ljust(5, "0")
            str_desvio_padrao = str((desvio_padrao[l, c] * 100).round(3)).ljust(5, "0")
            medida_media.iloc[l, c] = f"{str_media}% ± {str_desvio_padrao}%"

    return medida_media