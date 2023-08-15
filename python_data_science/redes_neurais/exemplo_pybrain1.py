# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:49:18 2023

@author: Luis
"""
import sys, numpy
sys.modules["scipy.random"] = numpy.random
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

rede = FeedForwardNetwork()

camadaEntrada = LinearLayer(2) # 2 neurônios 
camadaOculta = SigmoidLayer(3) # 3 neurônios
camadaSaida = SigmoidLayer(1) # 1 neurônio
bias1 = BiasUnit()
bias2 = BiasUnit()

rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

entradaOculta = FullConnection(camadaEntrada, camadaOculta)
ocultaSaida = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2, camadaSaida)

rede.sortModules()

print(rede)
# pesos
print(entradaOculta.params) 
print(ocultaSaida.params)
print(biasOculta.params)
print(biasSaida.params)
