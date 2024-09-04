"""
Análise de Dados e Projeto de um Classiﬁcador
Objetivo Principal: Desenvolver classiﬁcadores baseados em distância e o k-NN para o conjunto
de dados “Dry Bean” da UCI. https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
Inicialmente deve-se analisar os dados, realizar o pré-processamento, para posteriormente
projetar e analisar os classiﬁcadores
"""
# Importação das bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import mutual_info_classif
from scipy.spatial import distance
from imblearn.over_sampling import SMOTE


def load_data(data_id):
    """ Carrega um conjunto de dados a partir do repositório UCI.

    Esta função recupera o conjunto de dados associados ao `data_id` fornecido e utiliza
    a função `fetch_ucirepo` para extrair os `feaures`, os `targets`, as informações dos
    metadados e as variáveis do conjunto de dados do repositório UCI, e então separa as
    variáveis de entrada (X) da variável de saída (y). Ambas as variáveis são retornadas
    como DataFrames do Pandas.
    Args:
        data_id (int): O ID do conjunto de dados no repositório UCI.

    Returns:
        tuple: Uma tupla contendo:
            - x (array-like): As características do conjunto de dados.
            - y (array-like): Os alvos correspondentes.
            - metadata (dict): Os metadados associados ao conjunto de dados.
            - variables (dict): As informações sobre as variáveis no conjunto de dados.
    """

    repository_dataset = fetch_ucirepo(id=data_id)
    x = repository_dataset.data.features
    y = repository_dataset.data.targets
    metadata = repository_dataset.metadata # metadata
    variables = repository_dataset.variables #variable information

    return x, y, metadata, variables


def pre_process(x, y, test_size, random_state):
    """ Preprocessa os dados de entrada, dividindo-os em conjuntos de treino e teste, normalizando
    as características, balanceando o conjunto de treino usando a técnica SMOTE e codificando os
    rótulos.

    Args:
        x (array-like): As aracterísticas do conjunto de dados.
        y (array-like): Os alvos correspondentes..
        test_size (float): Proporção dos dados que será usada como conjunto de teste.
        random_state (int): Semente de aleatoriedade para garantir reprodutibilidade na divisão dos
        dados e no balanceamento.

    Returns:
        x_train_bal (array-like): Conjunto de treino balanceado e normalizado.
        x_test (array-like): Conjunto de teste normalizado.
        y_train_bal (array-like): Rótulos do conjunto de treino balanceado.
        y_test (array-like): Rótulos do conjunto de teste.
    """
    # Dividindo os dados em treino e teste (70% treino, 30% teste)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)
    # Iniciando o scaler, aprendendo a normaização nos dados de treino e aplicando-a nos dados
    # de teste
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # Utilizando o SMOTE para balanceamento das classes
    x_train_bal, y_train_bal = SMOTE(random_state=random_state).fit_resample(x_train, y_train)
    # Codifinado as labels, transformando as classes em valores numéricos.
    label_encoder = LabelEncoder()
    y_train_bal = label_encoder.fit_transform(y_train_bal.values.ravel())
    y_test = label_encoder.transform(y_test.values.ravel())
    classes = label_encoder.inverse_transform(np.unique(y_train_bal))
    return x_train_bal, x_test, y_train_bal, y_test, classes


def heat_map(matrix, matrix_name):
    """ Gera e exibe um mapa de calor (heatmap) para a matriz fornecida.

    Args:
        matrix (array-like): A matriz de dados que será visualizada como um mapa de calor.
        matrix_name (str): O título que será exibido no topo do mapa de calor.
    """
    plt.figure(figsize=(12,8))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title(f"{matrix_name}")
    plt.show()


def disp_matriz(conf_matrix, class_name, classification):
    """ Exibe a matriz de confusão fornecida utilizando uma visualização gráfica.

    Args:
        conf_matrix (array-like): A matriz de confusão que será exibida. Cada célula da matriz
        representa a contagem de classificações corretas e incorretas para cada classe.
        class_name (list de str): Os nomes das classes a serem exibidos nos rótulos da matriz
        de confusão.
    """
    _, ax = plt.subplots(figsize=(12,8))
    plt.title(f"Matriz Confusão - {classification}")
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_name)
    disp.plot(ax=ax, cmap=plt.cm.Oranges) # pylint: disable=no-member
    ax.set_ylabel('Classe Real', fontsize=14)
    plt.show()


# # Função para calcular as correlações e retornar as top n features
def top_n_features(x_train_bal, y_train_bal, x_test, method, n):
    """ Seleciona as principais características de um conjunto de dados de treinamento balanceado 
    com base em diferentes métodos de seleção de características.

    Args:
   
        x_train_bal (array-like): Conjunto de treino balanceado e normalizado.
        y_train_bal (array-like): Rótulos do conjunto de treino balanceado.
        x_test (array-like): Conjunto de teste normalizado.
        method (str): O método de seleção de características a ser usado. As opções incluem:
            - 'correlation': Seleciona as características com maior correlação absoluta com
            a variável de resposta.
            - 'mutual_info': Seleciona as características com maior informação mútua em relação
            à variável de resposta.
            - 'fisher': Seleciona as características com base nos coeficientes absolutos de um
            Analisador Discriminante Linear (LDA).
            - 'pca': Seleciona componentes principais usando Análise de Componentes Principais
            (PCA).
        n (int): Número de características principais a serem selecionadas.
           
    Return:
        top_features (array-like): Índices ou componentes principais das características mais
        importantes, dependendo do método utilizado:
            - Se 'correlation', 'mutual_info' ou 'fisher', retorna os índices das características
            principais.
            - Se 'pca', retorna os componentes principais (vetores de pesos das características
            originais).
    """

    if method == 'correlation':
        correlation_matrix = pd.DataFrame(x_train_bal).corr()
        heat_map(correlation_matrix, "Matriz de Correlação")
        correlations = pd.Series(np.corrcoef(x_train_bal.T, y_train_bal)[-1, :-1])
        top_features = correlations.abs().sort_values(ascending=False).head(n).index
        x_train = x_train_bal[:, top_features]
        x_test = x_test[:, top_features]
    elif method == 'mutual_info':
        mutual_info = mutual_info_classif(x_train_bal, y_train_bal)
        top_features = np.argsort(mutual_info)[-n:]
        x_train = x_train_bal[:, top_features]
        x_test = x_test[:, top_features]
    elif method == 'fisher':
        lda = LinearDiscriminantAnalysis()
        lda.fit(x_train_bal, y_train_bal)
        fisher_ratios = np.sum(np.abs(lda.coef_), axis=0)
        top_features = np.argsort(fisher_ratios)[-n:]
        x_train = x_train_bal[:, top_features]
        x_test = x_test[:, top_features]
    elif method == 'pca':
        pca = PCA(n_components=n)
        pca.fit(x_train_bal)
        x_train = pca.transform(x_train_bal)
        x_test = pca.transform(x_test)
        top_features = pca.components_
    return top_features, x_train, x_test


def mahalanobis_classifier(x_train_bal, x_test, y_train_bal, y_test):
    """ Implementa um classificador baseado na distância de Mahalanobis utilizando os dados de
    treino balanceados, realiza previsões nos dados de teste e avalia o desempenho do modelo.  

    Args:
        x_train (array-like): Dados de treino usados para treinar o modelo.
        x_test (array-like): Dados de teste usados para avaliar o modelo.
        y_train (array-like): Rótulos das classes correspondentes aos dados de treino balanceados.
        y_test (array-like): Rótulos das classes correspondentes aos dados de teste.

    Return:
        metrics (list): Uma lista contendo as seguintes métricas de desempenho:
            - accuracy (float): A precisão do modelo, calculada como a proporção de previsões
            corretas.
            - sensitivity (float): A sensibilidade do modelo, calculada como a média das taxas de
            verdadeiros positivos por classe.
            - specificity (float): A especificidade do modelo, calculada como a média das taxas de
            verdadeiros negativos por classe.
        conf_matrix (array-like): A matriz de confusão, que mostra o desempenho do modelo em termos
        de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos.
                
    Notas:
        A função calcula a distância de Mahalanobis entre cada amostra de teste e a média 
        de cada classe no conjunto de treino, utilizando a matriz de covariância do conjunto 
        e treino. A classe com a menor distância é atribuída à amostra.
    """
    cov_matrix = np.cov(x_train_bal, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    y_pred = []
    classes = np.unique(y_train_bal)
    for x in x_test:
        distances = []
        for c in classes:
            mean = np.mean(x_train_bal[y_train_bal == c], axis=0)
            distances.append(distance.mahalanobis(x, mean, inv_cov_matrix))
        y_pred.append(np.argmin(distances))
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sensitivity = (np.diag(conf_matrix) / conf_matrix.sum(axis=1)).mean()
    caracteristic = (np.diag(conf_matrix) / conf_matrix.sum(axis=0)).mean()
    return [accuracy, sensitivity, caracteristic], conf_matrix

def knn_classifier(x_train_bal, x_test, y_train_bal, y_test, k):
    """ Treina um classificador k-Nearest Neighbors (k-NN) usando os dados de treino balanceados,
    realiza previsões nos dados de teste e avalia o desempenho do modelo.  

    Args:
        x_train (array-like): Dados de treino usados para treinar o modelo.
        x_test (array-like): Dados de teste usados para avaliar o modelo.
        y_train (array-like): Rótulos das classes correspondentes aos dados de treino balanceados.
        y_test (array-like): Rótulos das classes correspondentes aos dados de teste.
        k (int): Número de vizinhos mais próximos a serem considerados pelo classificador k-NN.

    Return:
        metrics (list): Uma lista contendo as seguintes métricas de desempenho:
            - accuracy (float): A precisão do modelo, calculada como a proporção de previsões
            corretas.
            - sensitivity (float): A sensibilidade do modelo, calculada como a média das taxas de
            verdadeiros positivos por classe.
            - specificity (float): A especificidade do modelo, calculada como a média das taxas de
            verdadeiros negativos por classe.
        conf_matrix (array-like): A matriz de confusão, que mostra o desempenho do modelo em termos
        de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos.
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_bal, y_train_bal)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sensitivity = (np.diag(conf_matrix) / conf_matrix.sum(axis=1)).mean()
    caracteristic = (np.diag(conf_matrix) / conf_matrix.sum(axis=0)).mean()
    return [accuracy, sensitivity, caracteristic], conf_matrix


def lda_classifier(x_train, x_test, y_train, y_test):
    """ Treina um classificador Linear Discriminant Analysis (LDA) usando os dados de treino
    balanceados, realiza previsões nos dados de teste e avalia o desempenho do modelo.

    Args:
        x_train (array-like): Dados de treino usados para treinar o modelo.
        x_test (array-like): Dados de teste usados para avaliar o modelo.
        y_train (array-like): Rótulos das classes correspondentes aos dados de treino balanceados.
        y_test (array-like): Rótulos das classes correspondentes aos dados de teste.

    Returns:
        metrics (list): Uma lista contendo as seguintes métricas de desempenho:
            - accuracy (float): A precisão do modelo, calculada como a proporção de previsões
            corretas.
            - sensitivity (float): A sensibilidade do modelo, calculada como a média das taxas de
            verdadeiros positivos por classe.
            - specificity (float): A especificidade do modelo, calculada como a média das taxas de
            verdadeiros negativos por classe.
        conf_matrix (array-like): A matriz de confusão, que mostra o desempenho do modelo em termos
        de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos.
    """
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    y_pred = lda.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sensitivity = (np.diag(conf_matrix) / conf_matrix.sum(axis=1)).mean()
    caracteristic = (np.diag(conf_matrix) / conf_matrix.sum(axis=0)).mean()
    return [accuracy, sensitivity, caracteristic], conf_matrix

def project_classifier(features, x_train, x_test, y_train, y_test, classes, k):
    """ Executa a classificação usando três algoritmos (Mahalanobis, k-NN, e LDA) e suas variantes
    com diferentes conjuntos de características. Calcula e retorna as métricas de precisão e 
    sensibilidade para cada classificador.

    Args:
    features (dict): Um dicionário contendo os subconjuntos de características pré-processadas.
        As chaves são os nomes das técnicas de seleção de características (e.g., 'corr', 'mutual', 
        'fisher', 'pca') e os valores são tuplas contendo o conjunto de características
        pré-processado para treinamento e teste.
    x_train (array-like): Conjunto de dados de treinamento para os classificadores sem
    pré-processamento adicional.
    x_test (array-like): Conjunto de dados de teste para os classificadores sem pré-processamento
    adicional.
    y_train (array-like): Rótulos de classe correspondentes aos dados de treinamento.
    y_test (array-like): Rótulos de classe correspondentes aos dados de teste.
    classes (list): Lista contendo os nomes das classes para a visualização da matriz de confusão.
    k (int): O número de vizinhos a serem considerados no classificador k-NN.

    Returns:
    -------
    projetos (pandas.DataFrame): Um DataFrame contendo as métricas de precisão e sensibilidade para
    cada classificador, bem como para as suas variantes que utilizam diferentes subconjuntos de
    características.
    """
    projetos = pd.DataFrame(columns=['accuracy', 'sensitivity', 'caracteristic'],
                            index=['mahalanobis','mahalanobis_corr', 'mahalanobis_mutual',
                            'mahalanobis_fisher', 'mahalanobis_pca', 'knn','knn_corr',
                            'knn_mutual','knn_fisher', 'knn_pca', 'lda','lda_corr',
                            'lda_mutual','lda_fisher', 'lda_pca'])

    classificadores = ['mahalanobis', 'knn', 'lda']
    for classificador in classificadores:
        if classificador == 'mahalanobis':
            projetos.loc[classificador], conf_matrix = mahalanobis_classifier(x_train,
                                                                x_test, y_train, y_test)
            disp_matriz(conf_matrix, classes, classificador)
            for feature in features:
                projetos.loc[f'{classificador}_{feature}'], conf_matrix = mahalanobis_classifier(
                    features[feature][1], features[feature][2], y_train, y_test)
                disp_matriz(conf_matrix, classes, f'{classificador}_{feature}')
            print(projetos)
        elif classificador == 'knn':
            projetos.loc[classificador], conf_matrix = knn_classifier(x_train,
                                                                x_test, y_train, y_test, k)
            disp_matriz(conf_matrix, classes, classificador)
            for feature in features:
                projetos.loc[f'{classificador}_{feature}'], conf_matrix = knn_classifier(
                    features[feature][1], features[feature][2], y_train, y_test, k)
                disp_matriz(conf_matrix, classes, f'{classificador}_{feature}')
            print(projetos)
        elif classificador == 'lda':
            projetos.loc[classificador], conf_matrix = lda_classifier(x_train,
                                                                x_test, y_train, y_test)
            disp_matriz(conf_matrix, classes, classificador)
            for feature in features:
                projetos.loc[f'{classificador}_{feature}'], conf_matrix = lda_classifier(
                    features[feature][1], features[feature][2], y_train, y_test)
                disp_matriz(conf_matrix, classes, f'{classificador}_{feature}')
            print(projetos)
        else:
            pass

    return projetos


# # Função principal para execução do fluxo
def main():
    """ Carregamento e pré-processamento dos dados """

    data_id = 602 # Repositório UCI - Dry Bean
    test_size = 0.3 # Conjunto de teste = 30%
    random_state = 42 # Semente para o gerador de números randômicos, garantindo a reprodutibilidade
    qte = 5 # Quantidades de parâmetros a ser utilizado
    k = 10 # Número de visinhos mais próximos
    x, y, metadata, variables = load_data(data_id)
    x_train, x_test, y_train, y_test, classes = pre_process(x, y, test_size, random_state)

    top_features_indices = {
        'corr': top_n_features(x_train, y_train, x_test, method='correlation', n=qte),
        'mutual': top_n_features(x_train, y_train, x_test, method='mutual_info', n=qte),
        'fisher': top_n_features(x_train, y_train, x_test, method='fisher', n=qte),
        'pca': top_n_features(x_train, y_train, x_test, method='pca', n=qte)
    }

    result = project_classifier(top_features_indices, x_train, x_test, y_train, y_test, classes, k)

    print(f"""Com base na \033[1;31macurâcia\033[0m o melhor projeto de classificador é o \033[1;33m
        {result['accuracy'].idxmax()}\033[0m com valor de: \033[1;33m{result['accuracy'].max():.3%}
        \033[0m""")
    print(f"""Com base na \033[1;31msensibilidade\033[0m o melhor projeto de classificador é o
          033[1;33m{result['sensitivity'].idxmax()}\033[0m com valor de: \033[1;33m{result[
              'sensitivity'].max():.3%}\033[0m""")
    print(f"""Com base na \033[1;31mespecificidades\033[0m o melhor projeto de classificador é o
          \033[1;33m{result['caracteristic'].idxmax()}\033[0m com valor de: \033[1;33m{result[
              'caracteristic'].max():.3%}\033[0m""")

if __name__ == "__main__":
    main()
