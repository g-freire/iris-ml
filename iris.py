#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# EFICIENCIA DE DIVERSOS MODELOS DE CLASSIFICACAO SUPERVISIONADA

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
from pandas.plotting import scatter_matrix
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

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"  opcao via url
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] # caracteristicas
dataset = pandas.read_csv('iris.csv', names=names)  # lendo dados

print("--------- IRIS DATASET ----------------")
# print(dataset.shape)
# print(dataset.head(20)) 	# printa 20 primeiros items
# print(dataset.describe())	# Estatisticas basicas
# print(dataset.groupby('class').size()) 	# distribuicao por classes ( 50 - 50 - 50)


# # Plotagem  box e whisker
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()
# # Plotagem histogramas  -- observar distribuicoes Gaussianas
# dataset.hist()
# plt.show()
#
# # scatter plot matrix -- note o agrupamento de alguns atributos pares, sugere alta correlacao e relacao previsivel
# scatter_matrix(dataset)
# plt.show()

# Criacao de Modelos 

# 1) Separacao do Dataset de Treinamento / Teste
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=.30, random_state=seed)

#2) Opcoes de teste e avalicao metricas
seed = 7
scoring = 'accuracy'


# 3) Algoritmo Spot Check 
print("*")
print("---------  Eficiencia de modelos de classificacão testados----------------")
print("*")

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
# fig = plt.figure()
# fig.suptitle('Comparacão Algoritmos')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
#plt.show()		descomentar para ver plot
print("*")
print("*")
print("---------  Métricas: Precisão da previsão, Matriz Confusão, Desempenho da Classifição ----------------")
print("*")

# 5) previsoes no dataset de validacao com tecnica vizinho mais proximo
print("---------  K-Vizinhos próximos (KN)  ----------------")
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# 6) previsoes no dataset de validacao com tecnica maquina de vetores suporte 
print("---------  Máquinas de Vetor Suporte (SVM)  ----------------")
svc = SVC()
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# 7) previsoes no dataset de validacao com tecnica  Naive Bayes Gaussiana
print("---------  Naive Bayes Gaussiana (GNB) ----------------")
gnv = GaussianNB()
gnv.fit(X_train, Y_train)
predictions = gnv.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

