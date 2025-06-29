import numpy as np

def store_design_variables(filename, var, numSails, NumSeg, dT, T, SegLength):
    with open(filename, 'w') as f:
        f.write(f'Design Variables for numSails = {numSails}, dT = {dT}, T = {T/365} Years, NumSeg = {NumSeg},  and SegLength = {SegLength}:\n')
        np.set_printoptions(threshold=np.inf) 
        f.write(np.array2string(var, separator=', '))

def generate_smooth_angles(num_angles, lb, ub, max_variation):
    angles = np.zeros(num_angles)
    angles[0] = np.random.uniform(lb[0], ub[0])
    for i in range(1, num_angles):
        min_angle = max(lb[i], angles[i-1] - max_variation)
        max_angle = min(ub[i], angles[i-1] + max_variation)
        angles[i] = np.random.uniform(min_angle, max_angle)
    return angles

def GeneticAlgorithm(S,P,K,TOL,G,dv,lb,ub,func, desHoverTime, constant_angles, T, w, TOLNEO,TOLEarth, max_variation, numSeg, dT, bodies, printStatements = 1):
    cost=np.ones((S,1))*1000
    Pi=np.empty((0,S))
    meanParents=[]
    Orig=np.zeros((G,S))
    Children=np.zeros((K,dv))
    Parents=np.zeros((P,dv))
    Lambda=np.zeros((S,dv))
    Gen=1
    start = 0
    
    # Generate starting population
    if max_variation == 0:
        pop_new=np.random.uniform(lb, ub, (S, dv))
    else:
        pop_new = np.zeros((S, dv))
        for i in range(S):
            pop_new[i, :3+numSeg] = np.random.uniform(lb[:3+numSeg], ub[:3+numSeg])
            pop_new[i, 3+numSeg:3+2*numSeg] = generate_smooth_angles(numSeg, lb[3+numSeg:], ub[3+numSeg:], max_variation)
            pop_new[i, 3+2*numSeg:3+3*numSeg] = generate_smooth_angles(numSeg, lb[3+2*numSeg:], ub[3+2*numSeg:], max_variation)

    pop_new[:, 0] = np.random.randint(lb[0], ub[0] + 1, S)  # Ensure degree is an integer
    

    while np.abs(np.min(cost))>TOL and Gen<G:
        if printStatements == 1:
            print("**********************************")
            print("Generation number : ", Gen)
        pop=pop_new

        #Evaluate population fitness
        for i in range(start, S):
            if printStatements == 1:
                print(f"String : {i+1}, Gen : {Gen} ")
            cost[i]=func(pop[i,:], desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies)

        #Sort population fitnesses
        Index=np.argsort(cost[:,0])
        pop=pop[Index,:]
        cost=cost[Index,:]

        if printStatements == 1:
            print(f"Best cost for generation {Gen} : {cost[0]}")

        #Select parents
        Parents=pop[0:P,:]
        meanParents.append(np.mean(cost[0:P]))


        #Generate K offspring
        for i in range(0,K,2):
            #Breeding parents
            alpha=np.random.uniform(0,1)
            beta=np.random.uniform(0,1)
            Children[i, :] = Parents[i, :] * alpha + Parents[i + 1, :] * (1 - alpha)
            Children[i + 1, :] = Parents[i, :] * beta + Parents[i + 1, :] * (1 - beta)
            
            # Ensure degree is an integer
            Children[i, 0] = int(np.round(Children[i, 0]))
            Children[i + 1, 0] = int(np.round(Children[i + 1, 0]))

            # Clip angles to ensure they are within bounds
            Children[i, :] = np.clip(Children[i, :], lb, ub)
            Children[i + 1, :] = np.clip(Children[i + 1, :], lb, ub)

        
        #Overwrite population with P parents, K children, and S-P-K random values
        random_values = np.random.uniform(lb, ub, (S - P - K, dv))
        random_values[:, 0] = np.random.randint(lb[0], ub[0] + 1, S - P - K)  # Ensure degree is an integer
        pop_new = np.vstack((Parents, Children, random_values))

        #Store costs and indices for each generation
        Pi= np.vstack((Pi, cost.T))
        Orig[Gen,:]=Index
        #Increment generation counter
        Gen=Gen+1
        start = P

    #Store best population 
    Lambda=pop    
    meanPi=np.mean(Pi,axis=1)
    minPi=np.min(Pi,axis=1)
    return Lambda, Pi, Orig, meanPi, minPi,meanParents,cost



