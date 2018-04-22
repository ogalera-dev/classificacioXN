# Xarxa neuronal

# Importar les dades
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# Codificar les variables categoriques
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Dividir les dades en entrenament i test.
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Escalar factors per evitar que alguna variable independent
# domini a la resta.
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Creacio de la xarxa
# install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
model = h2o.deeplearning(y = 'Exited',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(5,5),
                         epochs = 100,
                         train_samples_per_iteration = -2)

# Preedir els resultats.
y_pred = h2o.predict(model, newdata = as.h2o(test_set[-11]))
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)

# Matriu de confusio
cm = table(test_set[, 11], y_pred)

h2o.shutdown()