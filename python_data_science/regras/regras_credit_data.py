# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:34:43 2023

@author: Luis
"""

import Orange

base = Orange.data.Table('credit_data.csv')
base.domain

base_dividida = Orange.evaluation.testing.sample(base, n=0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(base_treinamento)

for regra in classificador.rule_list:
    print(regra)
    
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [classificador])
print(Orange.evaluation.CA(resultado))
