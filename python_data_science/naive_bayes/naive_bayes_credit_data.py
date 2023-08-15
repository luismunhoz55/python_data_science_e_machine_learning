 # -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:29:39 2023

@author: Luis
"""

import pandas as pd
import numpy as np

# importar a base de dados
base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92 # corrigir idades negativas

               
# Separar previsores da classe
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values


# Tratar valores faltantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])


# Transformar todos os atributos na mesma escala
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


# Dividir a base de dados entre treinamento e teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)


from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento) # Treinar

previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

print("A precisão do algoritmo é de ",precisao * 100,"%") # Precisão de 93.8%





