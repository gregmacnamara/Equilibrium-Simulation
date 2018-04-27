import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import scipy.interpolate
import scipy.optimize
import sys
def sigma(z,params):
    qH,qL,cH,lam,delta = params
    return qH*z/(qH*z + qL*(1-z))
def sigmaInv(z,params):
    qH,qL,cH,lam,delta = params
    return qL*z/(qH*(1-z) + qL*(z))
def phi(z,params):
    qH,qL,cH,lam,delta = params
    return (1-qH)*z/((1-qH)*z + (1-qL)*(1-z))
def phiInv(z,params):
    qH,qL,cH,lam,delta = params
    return (1-qL)*z/((1-qH)*(1-z) + (1-qL)*(z))
def eta(z,params):
    qH,qL,cH,lam,delta = params
    return qH*z + qL*(1-z)
def nu(z,params):
    qH,qL,cH,lam,delta = params
    return (1-qH)*z + (1-qL)*(1-z)


def simulateOptIndices(qH,cH,qL,delta,lam, pureRegime,T):
    '''calculating the threhsold assuming that cH is optimal every where it is optimal at beliefs above the next periods threhsold'''
    '''This assumption is right someimtes and wrong someitmes - mainly it's right if qL-cH>lam or delta = 1'''
    def calcBreakpoints(t,params,optThresholds,breakpoints,breakpointValues, pureRegime):
        '''We know that the value function is always piecewise lienar. Therefore, the only values we need to save to efficiently
        calculate the value function at an arbitraty belief is the points at which the policy changes -
        this function calculates these points'''
        if t==0:
            return [0,optLast(params,optThresholds,breakpoints,breakpointValues, pureRegime),1]
        else:
            nextBreakpoints = breakpoints[params][t-1]
            threshold = optThresholds[params][t]
            currentBreakpoints =[0,threshold,1]
            roundedSet = set()
            for x in nextBreakpoints:
                #DO I NEED BOTH?
                if sigmaInv(x,params) >= threshold and round(sigmaInv(x,params),10)<.99 and round(sigmaInv(x,params),10) not in roundedSet:
                    currentBreakpoints.append(sigmaInv(x,params))
                    roundedSet.add(round(sigmaInv(x,params),10))
                if phiInv(x,params)>= threshold and round(sigmaInv(x,params),10)<.99 and round(phiInv(x,params),10) not in roundedSet:
                    currentBreakpoints.append(phiInv(x,params))
                    roundedSet.add(round(phiInv(x,params),10))
            return currentBreakpoints

    def calcThreshold(t,params,optThresholds,breakpoints,breakpointValues, pureRegime):
        '''this function finds the point at which the derivative of changing the threshold is equal to 0.
        Thus it finds the optimal threshold given the calculation of the derivative'''
        if t==0:
            return optLast(params)
        if t>0:
            try:
                return sp.optimize.brentq(derivativeOfObjective,0,optThresholds[params][t-1],args = (params,t,optThresholds,breakpoints,breakpointValues,pureRegime))
            except ValueError:
                if optThresholds[params][t-1]>0:
                    return optThresholds[params][t-1]
                else:
                    return 0
    #             print(t)
    #             print(derivativeOfObjective(0,params,t,optThresholds,breakpoints,breakpointValues))
    #             print(derivativeOfObjective(optThresholds[params][t-1],params,t,optThresholds,breakpoints,breakpointValues))
    #             sys.exit(1)


    def derivativeOfObjective(x,params,t,optThresholds,breakpoints,breakpointValues, pureRegime):
        '''this uses our analytical results to characterize the derivative of teh objective with respect to the threshold in any given period'''
        qH,qL,cH,lam,delta = params
    #     If we want to use the derivative that is correct in the pure observational regime
        if pureRegime:
    #         print(1)
            return sum([delta**i for i in range(t+1)])*lam -eta(x,params)+cH-\
                    delta*eta(x,params)*U(params,t-1,sigma(x,params),optThresholds,breakpoints,breakpointValues,pureRegime)- \
                    delta*nu(x,params)*U(params,t-1,phi(x,params),optThresholds,breakpoints,breakpointValues,pureRegime)
        if qH-cH>lam>=qL:
    #         print(2)
            return sum([delta**i for i in range(t+1)])*lam -eta(x,params)+cH \
                        -delta*eta(x,params)*U(params,t-1,sigma(x,params),optThresholds,breakpoints,breakpointValues,pureRegime) \
                        -nu(x,params)*sum([delta**i for i in range(1,t+1)])*lam
        elif qH-cH>qL>lam:
    #         print(3)

            return (1-x/optThresholds[params][t-1])*sum([delta**i for i in range(t+1)])*qL \
                        + x/optThresholds[params][t-1]*(lam+delta*U(params,t-1,optThresholds[params][t-1],optThresholds,breakpoints,breakpointValues,pureRegime))  \
                        -eta(x,params)+cH -delta*eta(x,params)*U(params,t-1,sigma(x,params),optThresholds,breakpoints,breakpointValues,pureRegime ) \
                        -delta*nu(x,params)*U(params,t-1,phi(x,params),optThresholds,breakpoints,breakpointValues,pureRegime)

    def optLast(params,optThresholds,breakpoints,breakpointValues, pureRegime = False):
        '''simply calculates the known breakeven point in the last period given parameters'''
        qH,qL,cH,lam,delta = params

        if (qH-cH>lam>qL) or pureRegime:
    #         print ('pure')
            return (lam+cH-qL)/(qH-qL)
        elif qH-cH>qL>lam:
    #         print ('mixed')
            return cH/(qH-lam)

    def breakpointValue(params,t,optThresholds,breakpoints,breakpointValues, pureRegime):
        '''calculates and stores the value at each breakpoint to allow for linear interpolation'''
        qH,qL,cH,lam,delta = params
        pointsToCalculate = breakpoints[params][t]
        calculatedBreakpointValues = []
        for x in pointsToCalculate:
            if pureRegime or (qH-cH>lam>qL):
                if x>=optThresholds[params][t]:
                    calculatedBreakpointValues.append(eta(x,params)-cH + delta*eta(x,params)*U(params,t-1,sigma(x,params),optThresholds,breakpoints,breakpointValues,pureRegime) \
                                            + delta*nu(x,params)*U(params,t-1,phi(x,params),optThresholds,breakpoints,breakpointValues,pureRegime) )
                else:
                    calculatedBreakpointValues.append(lam+delta*U(params,t-1,x,optThresholds,breakpoints,breakpointValues,pureRegime))
            elif qH-cH>qL>lam:
                if t==0:
                    calculatedBreakpointValues.append(max( (1-x)*qL+x*qH -cH, (1-x)*qL+x*lam ))
                elif t>0:
                    if x>=optThresholds[params][t]:
                        result = eta(x,params)-cH + delta*eta(x,params)*U(params,t-1,sigma(x,params),optThresholds,breakpoints,breakpointValues,pureRegime) \
                                                + delta*nu(x,params)*U(params,t-1,phi(x,params),optThresholds,breakpoints,breakpointValues,pureRegime)
                        calculatedBreakpointValues.append(result)
                    else:
                        result = (1-x/optThresholds[params][t-1])*sum([qL*delta**i for i in range(t+1)] ) + \
                                    x/optThresholds[params][t-1]*(lam + delta*U(params,t-1,optThresholds[params][t-1],optThresholds,breakpoints,breakpointValues,pureRegime) )
                        calculatedBreakpointValues.append(result)
        return calculatedBreakpointValues

    def U(params,t,x,optThresholds,breakpoints,breakpointValues, pureRegime):
        '''the Buyers value function. It either returns 0 as a terminal condition or interpolates the value between known values of a breakpoint '''
        if t==-1:
            return 0
        else:
            try:
                return sp.interpolate.interp1d(breakpoints[params][t],breakpointValues[params][t],kind='linear')(x)/1
            except TypeError:
                print(x)
                sys.exit('Problem with U')
    '''simulate the system given the parameters '''
#     These will be dics that save everything


#     for lam in lams:
    params = qH,qL,cH,lam,delta


    mixedObservational= qH-cH>qL>lam
    pureObservational  = qH-cH>lam>qL
    #Initialize these as a remnant...
    breakpoints = dict()
    optThresholds = dict()
    breakpointValues = dict()

    breakpoints[params] = dict()
    breakpoints[params][0] = calcBreakpoints(0,params,optThresholds,breakpoints,breakpointValues, pureRegime)

    optThresholds[params] = {}
    optThresholds[params][0]= optLast(params,optThresholds,breakpoints,breakpointValues, pureRegime)
    breakpointValues[params] = dict()
    breakpointValues[params][0] = breakpointValue(params,0,optThresholds,breakpoints,breakpointValues, pureRegime)
#         print(breakpointValues)
    for t in np.arange(1,T,1):
#             print(t)
        if t in breakpointValues[params]:
            pass
        else:
            optThresholds[params][t] = calcThreshold(t,params,optThresholds,breakpoints,breakpointValues, pureRegime)
            breakpoints[params][t] = calcBreakpoints(t,params,optThresholds,breakpoints,breakpointValues, pureRegime)
            breakpointValues[params][t] = breakpointValue(params,t,optThresholds,breakpoints,breakpointValues, pureRegime)

    return optThresholds[params], breakpoints[params], breakpointValues[params]



'''THE REMAINING IS THE NECESSARY FUNCTIONS TO ANALYZE THE FULL SIMULATION'''
'''MANY OF THE FUNCTIONS REPLACE THE ABOVE'''
def equilibrium(qH,cH,qL,delta,lam,T):
    def optLast(params):
        '''return breakeven beleif in last period'''
        if qH-cH>lam>qL:
            return (lam+cH-qL)/(qH-qL)
        elif qH-cH>qL>lam:
            return cH/(qH-lam)
    def policyChange(params,t, policy1,policy2, belief,criticalBeliefsNextPeriod,criticalValuesNextPeriod,low,high):
        def calcDifference(belief):
            option1 =  calcValueBuyer(params,t, policy1, belief,criticalBeliefsNextPeriod,criticalValuesNextPeriod)
            option2 =  calcValueBuyer(params,t, policy2, belief,criticalBeliefsNextPeriod,criticalValuesNextPeriod)
            return option1-option2
        return sp.optimize.brentq(calcDifference,low,high)

    def calcValueBuyer(params,t, policy, belief,criticalBeliefsNextPeriod,criticalValuesNextPeriod):
        '''t = periods to go'''
        '''qH,qL,cH,lam,delta = params'''
        '''policy = tuple of price, belief conditional upon rejection'''
        '''criticalBeliefs is a dictionary of the breakpoints in policy for all future periods - needed for calculation of value in enxt period'''
        '''criticalValues is a dictionary of the values at each breakpoint in policy for all future periods - needed for calculation of value in enxt period'''
        qH,qL,cH,lam,delta = params
        price = policy[0]
        beliefCondRejection = policy[1]
        if price == cH:
            return eta(belief,params)-cH+delta*eta(belief,params)*U(params,t-1, sigma(belief,params),criticalBeliefsNextPeriod,criticalValuesNextPeriod)\
            + delta*nu(belief,params)*U(params,t-1, phi(belief,params),criticalBeliefsNextPeriod,criticalValuesNextPeriod)
        elif price <cH:
            if belief == 0:
                #Corresponds what happens when known
                return qL*sum([delta**i for i in range(t+1)])
            elif beliefCondRejection ==None:
                #Corresponds to offer of 0 where both types reject
                #The option where 0 is offered and everyone rejects
                return lam+delta*U(params,t-1,  belief,criticalBeliefsNextPeriod,criticalValuesNextPeriod)
            else:
                if belief>beliefCondRejection:
    #                 return lam+delta*U(params,t-1, belief,criticalBeliefsNextPeriod,criticalValuesNextPeriod)
                        return -100
                alpha = belief/beliefCondRejection
                return alpha* (lam+delta*U(params,t-1, beliefCondRejection,criticalBeliefsNextPeriod,criticalValuesNextPeriod) ) \
                    +(1-alpha)*(qL*sum([delta**i for i in range(t+1)])-price)


    def calcValueSeller(params,t, policy, belief,criticalBeliefs,criticalValues):
        '''qH,qL,cH,lam,delta = params'''
        '''t = periods to go'''
        '''policy = tuple of price, belief conditional upon rejection'''
        '''criticalBeliefs is a list of the breakpoints in policy in the next period - needed for calculation of value in enxt period'''
        '''criticalValues is a list of the values at each breakpoint in the next period - needed for calculation of value in enxt period
            make sure that this one in the Seller function is the list of the Seller's value '''
        qH,qL,cH,lam,delta = params
        price = policy[0]
        beliefCondRejection = policy[1]
        try:
            if policy == (cH,None):
                return cH + delta*qL*V(t-1, sigma(belief,params),criticalBeliefs,criticalValues) \
                    + delta*(1-qL)*V(t-1, phi(belief,params),criticalBeliefs,criticalValues)
            elif policy == (0,None):
                return delta*V(t-1,belief,criticalBeliefs,criticalValues)
            else:
                return max(price, delta*V(t-1,beliefCondRejection,criticalBeliefs,criticalValues))
        except:
                print(policy)
                sys.exit(1)
    def U(params,t,belief,criticalBeliefs,criticalValues):
        '''the Buyers value function. It either returns 0 as a terminal condition or interpolates the value between known values of a breakpoint '''
        if t==-1:
            return 0
        else:
            try:
                return sp.interpolate.interp1d(criticalBeliefs,criticalValues,kind='linear')(belief)/1
            except TypeError:
                print(criticalBeliefs,criticalValues,belief)
                sys.exit('Problem with U')

    def V(t,belief,criticalBeliefs,criticalValuesSeller):
        '''t = periods to go - int'''
        '''belief = current belief = float'''
        '''criticalBeliefs = list which forms basis of next periods values'''
        '''criticalValuesSeller = list which is value at each point in the basis for next period'''
        if t==-1:
            return 0
        else:
            if belief == 0:
                return 0
            if belief in criticalBeliefs:
                location = np.searchsorted(criticalBeliefs,belief, 'left')
                return criticalValuesSeller[location]
            else:
                location = np.searchsorted(criticalBeliefs,belief, 'right')
                return criticalValuesSeller[location]
#Period T
    params = qH,qL,cH,lam,delta
    optPrices = [0,cH]
    value = list()
    criticalBeliefs = dict()
    optPolicy = dict()
    criticalValues = dict()
    criticalValuesSeller=dict()

    # Terminal Conditions:
    criticalValues[-1] = []
    criticalValuesSeller[-1] = []
    optPolicy[-1] = []
    criticalBeliefs[-1] = []

    #Last Period: Offer 0 or CH depending on optLast Beliefs
    criticalBeliefs[0]= [0]
    optPolicy[0]= [(0,None)]

    criticalBeliefs[0].append(optLast(params))
    optPolicy[0].append((0,1))

    criticalBeliefs[0].append(1)
    optPolicy[0].append((cH,None))


    criticalValues[0] = [calcValueBuyer(params,0,policy,belief,criticalBeliefs[-1],criticalValues[-1]) for policy, belief in zip(optPolicy[0],criticalBeliefs[0])]


    criticalValuesSeller[0] = [calcValueSeller(params,0,policy,belief,criticalBeliefs[-1],criticalValuesSeller[-1]) for policy, belief in zip(optPolicy[0],criticalBeliefs[0])]
    #Period 0 is done
    #Period 1 - determine available prices and the beliefs that can be generated with them:
    for t in range(1,T+1):
        if t%50 == 0:
            print(t)
        potentialPolicies = [(delta*x[0],x[1] ) for x in zip(criticalValuesSeller[t-1],criticalBeliefs[t-1]) if x[1]>0 and delta*x[0]<cH]
        potentialPolicies.append((0,None))
        potentialPolicies.append((cH,None))


        # determine OptPolicy at each critical point. how to determine critical points?
        #create a grid. Determine when policy changes and then evaluate where it changes...problem is that within the grid,
        # it may change twice, but that seems unlikely
        searchGrid1 = np.linspace(0,criticalBeliefs[t-1][1],10)
        searchGrid2 = np.linspace(criticalBeliefs[t-1][1],1,1000)
        searchGrid = np.concatenate((searchGrid1, searchGrid2))
        optPolicyGrid = list()
        criticalGrid= list()
        for belief in searchGrid:
            currentPolicy = [0,0]
            currentValue = 0
            for policy in potentialPolicies:
                try:
                    policyValue  = calcValueBuyer(params,t, policy, belief,criticalBeliefs[t-1],criticalValues[t-1])
                except:
                    sys.exit()
                if policyValue>currentValue:
                    currentOptimal = policy
                    currentValue = policyValue
            optPolicyGrid.append(currentOptimal)
            criticalGrid.append(currentValue)

        searchGridSellerValue = list()
        for policy, belief in zip(optPolicyGrid,searchGrid):
            searchGridSellerValue.append( calcValueSeller(params,1,policy,belief,criticalBeliefs[t-1],criticalValuesSeller[t-1]) )

        #now can determine criticalBeliefs from where policy changes
        criticalBeliefs[t] = [0]
        optPolicy[t] = [(0,None)]
        for i in range(0,len(searchGridSellerValue)-1):
            if searchGridSellerValue[i] != searchGridSellerValue[i+1]:
                w = policyChange(params,t, optPolicyGrid[i],optPolicyGrid[i+1], belief,criticalBeliefs[t-1] ,criticalValues[t-1],searchGrid[i],searchGrid[i+1])
                criticalBeliefs[t].append(w)
                optPolicy[t].append(optPolicyGrid[i])
        criticalBeliefs[t].append(1)
        optPolicy[t].append((cH,None))
        criticalValues[t] = [calcValueBuyer(params,t,policy,belief,criticalBeliefs[t-1],criticalValues[t-1]) for policy, belief in zip(optPolicy[t],criticalBeliefs[t])]
        criticalValuesSeller[t] = [calcValueSeller(params,t,policy,belief,criticalBeliefs[t-1],criticalValuesSeller[t-1]) for policy, belief in zip(optPolicy[t],criticalBeliefs[t])]
    return criticalBeliefs, optPolicy, criticalValues,criticalValuesSeller
