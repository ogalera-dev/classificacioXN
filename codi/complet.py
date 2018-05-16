#Part 1. Carregar les llibreries i fer el preprocessament!
# ---
# ---

#Carregar llibreries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Treure columnes que no aporten informació i la columna resultat.
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificar les variables categòriques.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Dividim les dades en train (80%) i test (20%).
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Normalitzem les dades.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Construir la xarxa neuronal
# ---
# ---

# Importar la llibreria keras.
import keras
from keras.models import Sequential
# Capes totalment connectades.
from keras.layers import Dense

# Iniciar la xarxa.
classifier = Sequential()

# 1era capa (capa d'entrada), el nombre d'entades correspon al nombre de variables que tenim
# el nombre de sortides correspon al nombre d'entrades de la següent capa.
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# 2ona Capa (capa interna 1)
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# 3era capa (Capa de sortida), només tenim una sortida (0-1) i per tant aquesta capa només té una neurona.
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compilar la xarxa neuronal.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Entrenament!!!
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 50)

# Part 3 - Avaluació
# ---
# ---

# Fem la predicció
y_pred = classifier.predict(X_test)

#Què farà cada client???
# <= 0.5 Sii no abandona el banc.
# > 0.5 Sii abandona el banc.
y_pred = (y_pred > 0.5)

# Observem el resultats amb una matriu de confusió. Com sempre els FN són els pitjors cassos
# es diu que el client no abandona el banc i per tant no es pren cap mesura, però al final
# l'acaba abandonant.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)