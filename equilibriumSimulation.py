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
def simulateOptIndices(qH,cH,qL,delta,lam, pureRegime,T):
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







'''HERE IS STUFF TO CALCULATE GITTINS INDEX'''
def initialGittinsIndex(params,gamma,periodsToGo):
    qH,qL,cH,lam,delta = params
    np.seterr(divide='ignore', invalid='ignore')
    numStates, transitionMatrix, rewardArray, states  = createBasics(periodsToGo,gamma,params )
    invStates = {v['location']: k for k, v in states.items()}
    gittinsIndices, _, _ = GittinsIndex(numStates,transitionMatrix,rewardArray,delta)
    for k, v in states.items():
        v['Gittins'] = gittinsIndices[v['location']]
    return states[(0,0)]['Gittins']
def createBasics(periods,gamma,params): 
    qH,qL,cH,lam,delta = params
    numStates = int(periods*(periods+1)/2)+1 + 1
    
    #state = (numSucc,periods)
    states = dict()
    states['NA'] = dict()
    states['NA']['posterior'] = gamma
    states['NA']['location'] = 0
    states[(0,0)] = dict()
    states[(0,0)]['posterior'] = gamma
    states[(0,0)]['location'] = 1
    count = 2
    #determine posteriors at each state
    for i in range(1,periods):
        for j in range(i+1):
            states[i,j] = dict()
            try:
                states[i,j]['posterior'] = sigma(states[(i-1,j-1)]['posterior'],params)
            except KeyError:
                states[i,j]['posterior'] = phi(states[(i-1,j)]['posterior'],params)
            states[i,j]['location'] = count
            count+=1

    #determine transitinos between states
    transitionMatrix = np.zeros((numStates,numStates))
    
    for i in range(0,periods):
        for j in range(i+1):
            if i<periods-1:
                locationOrig = states[i,j]['location']
                locationSucc = states[i+1,j+1]['location']
                locationFail = states[i+1,j]['location']
                posterior = states[i,j]['posterior']
                transitionMatrix[locationOrig, locationSucc] = eta(posterior, params)
                transitionMatrix[locationOrig, locationFail] = 1-eta(posterior, params)
            elif i == periods-1:
                locationOrig = states[i,j]['location']
                transitionMatrix[locationOrig, -1] = 1
    
    rewardArray = np.zeros(numStates)
    for i in range(0,periods):
        for j in range(i+1):
            locationOrig = states[i,j]['location']
            posterior = states[i,j]['posterior']        
            rewardArray[locationOrig] =  eta(posterior, params)-cH
    transitionMatrix[0,1] = 1
    transitionMatrix[-1,-1]=1
    return numStates, transitionMatrix, rewardArray, states


# def recursionStep(k,Q,B,D,delta,S,C,alpha,P):
#     #This follows the algorithm more precisely...the enxt one uses vector multiplication and addition to speed up
#     h = np.zeros(numStates)
#     for a in S[k-1]:
#         h[a] = P[a,alpha[k-1]] - sum([Q[a,b]*P[b,alpha[k-1]] for b in C[k]])

#     lambdaUpdate = delta/(1-delta*h[alpha[k-1]])
#     # Q2,B2,D2,h2 = recursionStep(2,Q,B,D,h,delta,S,C,alpha)
#     QNew = np.zeros((numStates,numStates))
#     bNew = np.zeros(numStates)
#     dNew = np.zeros(numStates)
#     for a in S[k]:
#         bNew[a] = B[a]+lambdaUpdate*h[a]*B[alpha[k-1]]
#         dNew[a] = D[alpha[k-1]]-B[a]/bNew[a] *(D[alpha[k-1]] - D[a] )
#         for b in range(numStates):
#             if b == alpha[k-1]:
#                 QNew[a,b] = -lambdaUpdate*h[a]
#             else:
#                 QNew[a,b] = Q[a,b]+lambdaUpdate*h[a]*Q[alpha[k-1],b]     
#     return QNew, bNew, dNew

# def recursionStepFast(k,Q,B,D,delta,S,C,alpha,P):
#     h = np.zeros(numStates)
#     for a in S[k-1]:
#         h[a] = P[a,alpha[k-1]] - sum([Q[a,b]*P[b,alpha[k-1]] for b in C[k]])

#     lambdaUpdate = delta/(1-delta*h[alpha[k-1]])
#     # Q2,B2,D2,h2 = recursionStep(2,Q,B,D,h,delta,S,C,alpha)
#     QNew = np.zeros((numStates,numStates))
#     bNew = np.zeros(numStates)
#     dNew = np.zeros(numStates)
#     bNew[S[k]] = (B + lambdaUpdate*h*B[alpha[k-1]])[S[k]]
#     dNew[S[k]] = (D[alpha[k-1]]-np.divide(B,bNew) *(D[alpha[k-1]] - D))[S[k]]

#     for a in S[k]:
#         QNew[a,:] =  Q[a,:]+lambdaUpdate*h[a]*Q[alpha[k-1],:]  
#         QNew[a,alpha[k-1]] =  -lambdaUpdate*h[a]
#     return QNew, bNew, dNew

def recursionStepFaster(k,Q,B,D,delta,S,C,alpha,P,numStates):
    h = np.zeros(numStates)
    h[S[k-1]]= (P[:,alpha[k-1]] - np.dot(Q[:,C[k]],P[C[k],alpha[k-1]]))[S[k-1]]
    
    lambdaUpdate = delta/(1-delta*h[alpha[k-1]])
    # Q2,B2,D2,h2 = recursionStep(2,Q,B,D,h,delta,S,C,alpha)
    QNew = np.zeros((numStates,numStates))
    bNew = np.zeros(numStates)
    dNew = np.zeros(numStates)
    bNew[S[k]] = (B + lambdaUpdate*h*B[alpha[k-1]])[S[k]]
    dNew[S[k]] = (D[alpha[k-1]]-np.divide(B,bNew) *(D[alpha[k-1]] - D))[S[k]]
    h.shape = (numStates,1)
    QNew[S[k],:] = Q[S[k],:] +lambdaUpdate*np.dot(h[S[k]],[Q[alpha[k-1],:]])
    h.shape=(numStates)
    QNew[S[k],alpha[k-1]] = -lambdaUpdate*h[S[k]]
#     for a in S[k]:
#         QNew[a,:] =  Q[a,:]+lambdaUpdate*h[a]*Q[alpha[k-1],:]  
#         QNew[a,alpha[k-1]] =  -lambdaUpdate*h[a]
    return QNew, bNew, dNew
def GittinsIndex(numStates,transitionMatrix,rewardArray,delta):
    P = transitionMatrix
    Q = np.zeros((numStates,numStates))
    B = np.ones(numStates)
    D = rewardArray

    #Things that will be updated each period
    Lambda=np.zeros(numStates)
    S = [None]* (numStates+2)
    S[1] = np.arange(numStates) #statesRemaining
    C = [None]* (numStates+2) #statesWithGI Calculated Already
    alpha = [None] * (numStates+1)

    
    k=1
    alpha[1]= np.argmax(D)  #the best remaining state
    
    k=2
    C[k] = [np.argmax(D)]
    S[k] = np.delete(S[1], C[k])
    Lambda[np.argmax(D)] = D[np.argmax(D)]

    Q,B,D = recursionStepFaster(k,Q,B,D,delta,S,C,alpha,P,numStates)
    alpha[k]= np.argmax(D)  

    for k in range(3,numStates+1):
        C[k] = C[k-1][:]
        C[k].append(np.argmax(D))
        S[k] = np.delete(S[1], C[k])
        Lambda[np.argmax(D)] = D[np.argmax(D)]
        Q,B,D = recursionStepFaster(k,Q,B,D,delta,S,C,alpha,P,numStates)
        alpha[k]= np.argmax(D)  #the best remaining state
    
    k = numStates+1
    C[k] = C[k-1][:]
    C[k].append(np.argmax(D))
    S[k] = np.delete(S[1], C[k])
    Lambda[np.argmax(D)] = D[np.argmax(D)]
    return Lambda,S,C