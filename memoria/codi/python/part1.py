# Part 1 - Processament de dades

# Llibreries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar dades
dataset = pd.read_csv('Churn_Modelling.csv')

# Seleccio de les variables independents (VI) que influeixen
# en la decissio d'abandonar o no el banc.
#
# Les variables que s'exclouen son:
# - RowNumber [0]
# - CustomerId [1]
# - Surname [2]
X = dataset.iloc[:, 3:13].values
# Seleccionem la variable dependent
y = dataset.iloc[:, 13].values

# Codificar les variables categoriques
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
codificador_X_1 = LabelEncoder()
X[:, 1] = codificador_X_1.fit_transform(X[:, 1])
codificador_X_2 = LabelEncoder()
X[:, 2] = codificador_X_2.fit_transform(X[:, 2])

# Es crean variables dummy perque no hi ha ordre entre les variables
# categoriques
codificador = OneHotEncoder(categorical_features = [1])
X = codificador.fit_transform(X).toarray()
# S'exclou la primera variable [0] dummy per eviat la (dummy variable trap)
X = X[:, 1:]

# Partir les dades en:
# - training set: dades per efectuar l'entrenament de la xarxa (8000)
# - test set: dades per comprovar l'efectivitat de la xarxa (2000)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fer feature scaling per evitar que una (o mes) variable independent domini la resta.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
