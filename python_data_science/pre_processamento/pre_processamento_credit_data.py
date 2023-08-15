import pandas as pd
base = pd.read_csv('credit_data.csv') 
base.describe()
base.loc[base['age'] < 0]

# Apagar a coluna
# base.drop('age', 1, inplace=True)

# Apagar somente os registros com problema
#base.drop(base[base.age < 0].index, inplace=True)

# Preencher os valores manualmente

# Preencher os valores com a média
#base.mean()
#base['age'].mean()

#base['age'][base.age > 0].mean() # fazer a média sem contar os valores negativos
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()

pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# Valores faltantes

from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])


# Escalonamento de atributos

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)





