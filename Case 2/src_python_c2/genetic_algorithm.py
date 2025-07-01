import numpy as np

def generate_smooth_angles(num_angles, lb, ub, max_variation):
    angles = np.zeros(num_angles)
    angles[0] = np.random.uniform(lb[0], ub[0])
    for i in range(1, num_angles):
        min_angle = max(lb[i], angles[i-1] - max_variation)
        max_angle = min(ub[i], angles[i-1] + max_variation)
        angles[i] = np.random.uniform(min_angle, max_angle)
    return angles

def GeneticAlgorithmSolarEscape(S, P, K, TOL, G, dv, lb, ub, func, constant_angles, T, w, max_variation, numSeg, dT, bodies, numSails, printStatements=1):
    # Adjust bounds for all sails
    dv_total = dv * numSails  # Total dimensionality
    lb = np.tile(lb, numSails)
    ub = np.tile(ub, numSails)

    cost = np.ones((S, 1)) * 1000
    prev_cost = None
    Pi = np.empty((0, S))
    meanParents = []
    Orig = np.zeros((G, S))
    Children = np.zeros((K, dv_total))
    Parents = np.zeros((P, dv_total))
    Lambda = np.zeros((S, dv_total))
    Gen = 1
    start = 0

   
    # Generate starting population
    if max_variation == 0:
        pop_new = np.random.uniform(lb, ub, (S, dv_total))
    else:
        pop_new = np.zeros((S, dv_total))
        for i in range(S):
            for sail in range(numSails):
                start_idx = sail * dv
                pop_new[i, start_idx:start_idx + 3 + numSeg] = np.random.uniform(lb[:3 + numSeg], ub[:3 + numSeg])
                pop_new[i, start_idx + 3 + numSeg:start_idx + 3 + 2 * numSeg] = generate_smooth_angles(numSeg, lb[3 + numSeg:], ub[3 + numSeg:], max_variation)
                pop_new[i, start_idx + 3 + 2 * numSeg:start_idx + 3 + 3 * numSeg] = generate_smooth_angles(numSeg, lb[3 + 2 * numSeg:], ub[3 + 2 * numSeg:], max_variation)

    pop_new[:, 0::dv] = np.random.randint(lb[0], ub[0] + 1, (S, numSails))  # Ensure degree is an integer
        

    while np.abs(np.min(cost)) > TOL and Gen < G:
        if printStatements == 1:
            print("**********************************")
            print("Generation number : ", Gen)
        pop = pop_new

        # Evaluate population fitness
        for i in range(start, S):
            if printStatements == 1:
                print(f"String : {i+1}, Gen : {Gen}")
            cost[i] = func(pop[i, :], constant_angles, T, w, dT, bodies, numSails, dv, numSeg)

        # Sort population fitnesses
        Index = np.argsort(cost[:, 0])
        pop = pop[Index, :]
        cost = cost[Index, :]

        if printStatements == 1:
            print(f"Best cost for generation {Gen} : {cost[0]}")

        # Select parents
        Parents = pop[0:P, :]
        meanParents.append(np.mean(cost[0:P]))

        # Generate K offspring
        for i in range(0, K, 2):
            # Breeding parents
            alpha = np.random.uniform(0, 1)
            beta = np.random.uniform(0, 1)
            Children[i, :] = Parents[i, :] * alpha + Parents[i + 1, :] * (1 - alpha)
            Children[i + 1, :] = Parents[i, :] * beta + Parents[i + 1, :] * (1 - beta)

            # Ensure degree is an integer
            Children[i, 0::dv] = np.round(Children[i, 0::dv])
            Children[i + 1, 0::dv] = np.round(Children[i + 1, 0::dv])

            # Clip angles to ensure they are within bounds
            Children[i, :] = np.clip(Children[i, :], lb, ub)
            Children[i + 1, :] = np.clip(Children[i + 1, :], lb, ub)

        # Overwrite population with P parents, K children, and S-P-K random values
        random_values = np.random.uniform(lb, ub, (S - P - K, dv_total))
        random_values[:, 0::dv] = np.random.randint(lb[0], ub[0] + 1, (S - P - K, numSails))  # Ensure degree is an integer
        pop_new = np.vstack((Parents, Children, random_values))

        # Store costs and indices for each generation
        Pi = np.vstack((Pi, cost.T))
        Orig[Gen, :] = Index
        # Increment generation counter
        Gen += 1
        start = P

    # Store best population
    Lambda = pop
    meanPi = np.mean(Pi, axis=1)
    minPi = np.min(Pi, axis=1)
    return Lambda, Pi, Orig, meanPi, minPi, meanParents, cost


