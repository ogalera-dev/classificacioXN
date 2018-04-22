# Part 3 - Efectura les prediccions.

# Prediccions.
y_pred = xarxa.predict(X_test)
# si la prediccio d'abandonament es superior a 0.5 interpretem que el client acabara abandonant el banc
y_pred = (y_pred > 0.5) 

