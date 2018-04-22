#Part 4 - Comprobar els resultats.
from sklearn.metrics import confusion_matrix
#Per comprovar els resultats utilitzem una matriu de confusio
cm = confusion_matrix(y_test, y_pred)
