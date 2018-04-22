# XARXA NEURONAL

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

# Part 2 - Construir la xarxa neuronal

# Importing la llibreria Keras
import keras
from keras.models import Sequential #iniciar la xarxa
from keras.layers import Dense #constuir les capes de la xarxa

# Iniciar la xarxa
xarxa = Sequential()

# Afegir la capa d'entrada.
# - output_dim: nombre de sortides cap a la capa interna de neurones (nombre de neurones
# que tindra la capa interna, una bona aproximacio es (nNeuronesCapaEntrada+nNeuronesCapaSortida)/2.
# - init: com iniciar els pesos de les synapses.
# - activation: quina funcio d'activacio s'utilitza, en aquest cas, s'utilitzara
# una funcio de rectificiacio en totes les capes i una funcio sigmoide en la capa de sortida.
xarxa.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Afegir la capa interna.
xarxa.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Afegir la capa de sortida, aquesta utilitza una sigmoide com a funcio d'activacio i
# com volem obtenir un resultat binari (abandonara o no el banc) nomes cal una 
# neurona que produeixi 0 si no abandonara o 1 si abandonara.
xarxa.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compilar la xarxa neuronal.
# - optimiser: nom de l'algorisme per trobar els pessos optims en les synapses, s'utilitza
# el descens del gradient estocastic, i una implementacio molt eficient d'aquest algorisme
# es diu adam.
# - loss: Funcio de cost del descens del gradient.
# - metrics: Per dir si la xarxa es bona o no es te en compte la precissio dels resultats.
xarxa.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Entrenament de la xarxa.
# S'APLICA UNA TECNICA DE MICROBATCH (BATCH DE MIDA 10).
# - X_train: variables independents (explicativa).
# - y_train: variables dependnets (explicada).
# - batch_size: mida de cada batch a partir del qual s'aplica el backpropagation.
# - nb_epoch: nombre de epochs a processar abans d'acabar d'entrenar la xarxa.
xarxa.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Efectura les prediccions.

# Prediccions.
y_pred = xarxa.predict(X_test)
# si la prediccio d'abandonament es superior a 0.5 interpretem que el client acabara abandonant el banc
y_pred = (y_pred > 0.5) 

#Part 4 - Comprobar els resultats.
from sklearn.metrics import confusion_matrix
#Per comprovar els resultats utilitzem una matriu de confusio
cm = confusion_matrix(y_test, y_pred)
