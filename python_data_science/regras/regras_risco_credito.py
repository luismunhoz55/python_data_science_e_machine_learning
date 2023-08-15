# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:16:46 2023

@author: Luis
"""

import Orange

base = Orange.data.Table('risco_credito.csv')
base.domain

cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(base)
for regra in classificador.rule_list:
    print(regra)
    
resultado = classificador([['boa','alta','nenhuma','acima_35'],['ruim','alta','adequada','0_15']])
for i in resultado:
    print(base.domain.class_var.values[i])