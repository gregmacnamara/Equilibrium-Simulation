from __future__ import division

import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
def indifferentDelta(qL,cH,qH,lam):
	def valueDiff(delta,qL,cH,qH,lam):
		return (qH-qL)/(qL-lam) - (1+delta*qL -delta)/(delta**2*cH) 
	try:
		return opt.brentq(valueDiff, 0.01,1, args=(qL,cH,qH,lam))
	except:
		return 1
		    
        
def criticalDelta(qL,cH,qH,lam):
    def valueDiff(delta,qL,cH,qH,lam):
        # return lam*(1-delta)+delta*qH*lam - cH*(1-delta)-(qH-qL)*delta*qL
        return lam/(1-delta)*(1-delta*(1-qH))+delta/(1-delta)*qL*(qL-qH) - cH
    try:
        return opt.brentq(valueDiff, 0.01,.99, args=(qL,cH,qH,lam))
    except:
        return np.nan
        if valueDiff(0.0001,qL,cH,qH,lam)>0:
            return 0
        else:
            return 1

def comparingRegions():
    qH=.8
    cH=.05
    qL = .3
    delta = .9

    deltas = np.arange(0,1,.01)

    leftLambda = ((qH-qL)*cH-qL**2)/(-qL)
    lambdas = np.arange(0,qL,.0002)

    def patientAssump(delta,qL):
    	return delta+delta**2*qL-1
    minDelta = opt.brentq(patientAssump, 0,1, args=(qL))
    # indiffDeltas = [indifferentDelta(qL,cH,qH,lam) for lam in lambdas]
    criticalDeltas = [criticalDelta(qL,cH,qH,lam) for lam in lambdas]
    print criticalDeltas[0]

 
    plt.axvline(x=qL, linestyle ='-', c = 'k')
    # plt.plot(lambdas,indiffDeltas, 'k-' )
    plt.plot(lambdas,criticalDeltas, 'k-' )
    plt.axhline(y = minDelta, linestyle ='-', c = 'k')

    # plt.fill_between(np.arange(qL,qH-cH+.01,.01),0,1,alpha = .75,label = 'Regime 1', color = 'blue')
    # plt.fill_between(np.arange(qL,qH-cH+.01,.01),0,minDelta,alpha = .25,label = 'Regime 1 - Conj', color = 'blue')

    # plt.fill_between(lambdas,indiffDeltas,1,alpha = .75, label = 'Regime 2 - Prop 1.4', color = 'green')  
    # plt.fill_between(lambdas,np.minimum(indiffDeltas,minDelta), minDelta,alpha = .25, label = 'Regime 2 - Prop 1.4 - Conj', color = 'green')

    # plt.fill_between(lambdas,minDelta,np.maximum(indiffDeltas,minDelta) ,alpha = .75, label = 'Regime 2 - In Progress', color = 'red')
    # plt.fill_between(lambdas,0,indiffDeltas ,alpha = .75, label = 'Regime 2 - In Progress - Conj', color = 'red')    


    # plt.fill_between(np.arange(qL,qH-cH+.01,.01),minDelta,1,alpha = .75,label = 'Regime 1', color = 'blue')
    # plt.fill_between(np.arange(qL,qH-cH+.01,.01),0,minDelta,alpha = .25,label = 'Regime 1 - Conj', color = 'blue')

    # plt.fill_between(lambdas,np.maximum(indiffDeltas,minDelta),1,alpha = .75, label = 'Regime 2 - Prop 1.4', color = 'green')  
    # plt.fill_between(lambdas,np.minimum(indiffDeltas,minDelta), minDelta,alpha = .25, label = 'Regime 2 - Prop 1.4 - Conj', color = 'green')

    # plt.fill_between(lambdas,minDelta,np.maximum(indiffDeltas,minDelta) ,alpha = .75, label = 'Regime 2 - In Progress', color = 'red')
    # plt.fill_between(lambdas,0,np.minimum(indiffDeltas,minDelta) ,alpha = .25, label = 'Regime 2 - In Progress - Conj', color = 'red')

    plt.text(0.85*(qH-cH),0.6,r'$q_H = {}$'.format(qH))
    plt.text(0.85*(qH-cH),0.55,r'$q_L = {}$'.format(qL))
    plt.text(0.85*(qH-cH),0.50,r'$c_H = {}$'.format(cH))
    plt.plot(.15, .8,  'ro') 
    plt.plot(.22, .8,  color ='orange', marker = 'o') 
    plt.plot(.29, .8, 'yo' ) 
    plt.plot(.31, .8, 'go' ) 
    plt.plot(.38, .8, 'bo' ) 
    plt.plot(.45, .8,  color ='indigo', marker = 'o' ) 
    plt.axis([0, qH-cH, 0, 1])
    plt.ylabel(r'$\delta$')
    plt.xlabel(r'$\lambda$')
    # plt.legend()
    # plt.title('Solution Regions')
    plt.savefig('Solution_RegionsMixed_qH{}_qL{}_cH{}_delta{}'.format(qH,qL,cH,delta).replace('.','')+'.pdf',bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
	comparingRegions()