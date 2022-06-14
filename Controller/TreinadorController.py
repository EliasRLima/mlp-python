import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


def treinar(dataset):
    colunas = ['nome', 'x1', 'x2', 'x3', 'esperado']
    campos = []
    nomes = []
    peso = []
    for amostra in dataset:
        elementos = amostra.split()
        campos.append([elementos[1], elementos[2], elementos[3]])
        nomes.append([elementos[0]])
        peso.append([elementos[4]])
    data = pd.array(campos, dtype=str)
    X = data.iloc[:, 1:5]
    y = data.select_dtypes(include=[object])
    le = preprocessing.LabelEncoder()
    y = y.apply(le.fit_transform)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(X_train, y_train.values.ravel())
    predictions = mlp.predict(X_test)
    return predictions

def separar(dataset):
    separado = treinar(dataset)
    grupoA = {}
    grupoB = {}
    for resultado in separado:
        if resultado['value'] == 1:
            grupoA.append(resultado)
        else:
            grupoB.append(resultado)
    grupos = {grupoA, grupoB}
    return grupos


