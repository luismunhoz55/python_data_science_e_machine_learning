# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:55:28 2023

@author: Luis
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

base = pd.read_csv('census.csv')

# Transformar variaveis categoricas em numeros

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

labelencoder_previsores = LabelEncoder()

previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])

onehotencoder = ColumnTransformer(
    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1, 3, 5, 6, 7, 8, 9, 13])],
    # Leave the rest of the columns untouched
    remainder='passthrough'
)

previsores = onehotencoder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
