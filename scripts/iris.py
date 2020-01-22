#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Machine Learn - Iris Dataset

# Checando  bibliotecas
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

# Carregando bibliotecas
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"  
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv('iris.csv', names=names) # via url aumenta latencia

print("--------- IRIS DATASET ----------------")
# Shape
print(dataset.shape)
# head - 20 primeiros items
print(dataset.head(20))

# Estatisticas basicas
print(dataset.describe())

# distribuicao por classes ( 50 - 50 - 50)
print(dataset.groupby('class').size())

# Plotagem  box e whisker 
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.savefig('plots/box-whisker.png')
# Plotagem histogramas  -- observar distribuicoes Gaussianas
dataset.hist()
plt.savefig('plots/histogram.png')

# scatter plot matrix -- note o agrupamento de alguns atributos pares, sugere alta correlacao e relacao previsivel
scatter_matrix(dataset)
plt.savefig('plots/scatter-matrix.png')

# Criacao de Modelos 

# 1) Separacao do Dataset de Validacao
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#2) Opcoes de teste e avalicao metricas
seed = 7
scoring = 'accuracy'

# 3) Algoritmo Spot Check 
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# analisa cada modelo em turnos
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# 4) Plotando resultado comparacao dos algoritmos
fig = plt.figure()
fig.suptitle('Comparacao Algoritmos')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
# plt.show()
plt.savefig('plots/benchmark.png')

# 5) Fazendo previsoes no dataset de validacao
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
