import numpy as np

def coordinateDescent(var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies, func, max_iter=20, tolerance=150, m=0.2, printStatements = 1):
    cost = func(var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies)
    no_change = 0

    for j in range(max_iter):
        for i in range(1,len(var)):
            if no_change > tolerance:
                m = m * 0.5
                no_change = 0

            inc_var = np.copy(var)
            inc_var[i] = inc_var[i] + inc_var[i] * m
            dec_var = np.copy(var)
            dec_var[i] = dec_var[i] - dec_var[i] * m

            inc_var_cost = func(inc_var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies)
            dec_var_cost = func(dec_var, desHoverTime, constant_angles, T, w, TOLNEO, TOLEarth, dT, bodies)

            if cost > inc_var_cost:
                var = inc_var
                cost = inc_var_cost
                no_change = 0
            else:
                no_change += 1

            if cost > dec_var_cost:
                var = dec_var
                cost = dec_var_cost
                no_change = 0
            else:
                no_change += 1
            
            if printStatements == 1:
                print("var #:", i, "m", m, "new cost:", cost)

        if cost < 0:
            break

    return var, cost
