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
