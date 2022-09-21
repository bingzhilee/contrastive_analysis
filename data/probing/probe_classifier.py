# -*- coding: utf-8 -*-

import sys
import bloscpack as bp
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV as gsc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV as rsc
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


"probing classifier per pos per part of sentence with grid search for C values"

X0_name = sys.argv[1]
Y0_name = sys.argv[2]
cat = sys.argv[3]
out = sys.argv[4]

X0_data = bp.unpack_ndarray_from_file(X0_name)
Y0_data = bp.unpack_ndarray_from_file(Y0_name)
X0_train, X0_test, Y0_train, Y0_test = train_test_split(X0_data,Y0_data, test_size=0.2, random_state=42,stratify=Y0_data)
print(cat, X0_train.shape[0])

# imbalanced data with 65% singular, so add parameter "class_weight"
model = LogisticRegression(class_weight = 'balanced',random_state=42,max_iter=1000)

# defining hyper-parameter range
params = {"C": [0.0001,0.001,0.01,0.05,0.1,1]}
repr_clf = gsc(model,params,n_jobs=1,refit = True)

repr_clf.fit(X0_train,Y0_train)
print(repr_clf.best_estimator_)

# do prediction on test set
y0_pred= repr_clf.predict(X0_test)
accu = round(accuracy_score(Y0_test,y0_pred)*100,1)
print(cat,accu)
#print(classification_report(Y0_test,y0_pred,digits=2))

#f_name = open("obj-pp_"+out+"_accuracy_PoS.txt","a")
#f_name.write(" ".join((cat,str(X0_train.shape[0]),str(X0_test.shape[0])))+"\n")
#f_name.write("accuracy=" + str(accu) +"\n")
#f_name.write("\n")
#f_name.close()
