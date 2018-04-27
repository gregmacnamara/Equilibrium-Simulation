'''Implementation of Nino-Mora Fast Gittins Index Calculation for my specific case of two arms, Bernoulli rewards, finite horizon, '''
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