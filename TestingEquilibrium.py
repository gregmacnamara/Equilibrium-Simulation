
# coding: utf-8

# In[63]:


import numpy as np
import scipy as sp
import scipy.interpolate
import sys
from matplotlib import pyplot as plt
import equilibriumSimulation as eqSim
import importlib
import pickle



if __name__ == '__main__':
	qH=.8
	cH=.4
	qL = .3
	delta = .9
	lam = .2
	T = 10000
	params = qH,qL,cH,lam,delta
	equilibriumResults = eqSim.equilibrium(qH,cH,qL,delta,lam,T)
	# criticalBeliefs, optPolicy, criticalValues,criticalValuesSeller = equilibriumResults
	file_pi = open('equilibriumResults.obj', 'w')
	pickle.dump(equilibriumResults, file_pi)
	print('Eq. Done')
	breakpointsDict=dict()
	optThresholdsDict = dict()
	breakpointValuesDict = dict()
	thresholdResults = eqSim.simulateOptIndices(qH,cH,qL,delta,lam, True,T+1)    
	# optThresholdsDict[params], breakpointsDict[params], breakpointValuesDict[params] =thresholdResults
	file_pi = open('thresholdResults.obj', 'w')
	pickle.dump(thresholdResults, file_pi)


