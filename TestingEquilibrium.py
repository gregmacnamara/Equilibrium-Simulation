
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
	# delta = .9
	# lam = .2
	T = 100
	lams = np.linspace(0,qH-cH-.05,10)
	deltas = np.linspace(.05,1,10)
	equilibriumResults = dict()
	for lam in lams:
		for delta in deltas:
			print(lam,delta)
			params = qH,qL,cH,lam,delta
			equilibriumResults[params] = eqSim.equilibrium(qH,cH,qL,delta,lam,T)
			# criticalBeliefs, optPolicy, criticalValues,criticalValuesSeller = equilibriumResults
			file_pi = open('equilibriumResults_Grid_T100.pickle', 'w')
			pickle.dump(equilibriumResults, file_pi)
	# breakpointsDict=dict()
	# optThresholdsDict = dict()
	# breakpointValuesDict = dict()
	# thresholdResults = eqSim.simulateOptIndices(qH,cH,qL,delta,lam, True,T+1)    
	# # optThresholdsDict[params], breakpointsDict[params], breakpointValuesDict[params] =thresholdResults
	# file_pi = open('thresholdResults.obj', 'w')
	# pickle.dump(thresholdResults, file_pi)


