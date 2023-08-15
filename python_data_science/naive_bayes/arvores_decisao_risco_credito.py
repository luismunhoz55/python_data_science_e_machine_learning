# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:25:16 2023

@author: Luis
"""

# Importar a base de dados
import pandas as pd

base = pd.read_csv('risco_credito.csv')

# Separar previsores da classe
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

# Transformar as variáveis categoricas em numeros
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()

previsores[:, 0] = labelEncoder.fit_transform(previsores[:,0])
previsores[:, 1] = labelEncoder.fit_transform(previsores[:,1])
previsores[:, 2] = labelEncoder.fit_transform(previsores[:,2])
previsores[:, 3] = labelEncoder.fit_transform(previsores[:,3])

# Treinar
from sklearn.tree import DecisionTreeClassifier, export_graphviz
classificador = DecisionTreeClassifier(criterion='entropy')
classificador.fit(previsores, classe)
print(classificador.feature_importances_)

export_graphviz(classificador,
                       out_file = 'arvore.dot',
                       feature_names = ['historia', 'divida', 'garantias', 'renda'],
                       class_names = ['alto', 'moderado', 'baixo'],
                       filled = True,
                       leaves_parallel = True)

# historia boa, divida alta, garantias nenhuma, renda > 35k
resultado = classificador.predict([[0, 0, 1, 2],[1, 1, 0, 0]]) # Risco baixo

print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)
