import numpy as np
import gc

def coordinate_descent_SolarEscape(var, constant_angles, T, w, dT, bodies, numSails, dv, NumSeg, lb, ub, func, max_iter=10, tolerance=150*20, m=0.2, printStatements=1):
    cost = func(var, constant_angles, T, w, dT, bodies, numSails, dv, NumSeg)
    no_change = 0

    for j in range(max_iter):
        for i in range(1, len(var)):
            if no_change > tolerance:
                m = m * 0.5
                no_change = 0
            if i % dv != 0:
                original_value = var[i]

                # Increment var[i] and calculate cost
                var[i] = original_value + original_value * m
                inc_var_cost = func(var, constant_angles, T, w, dT, bodies, numSails, dv, NumSeg)

                # Decrement var[i] and calculate cost
                var[i] = original_value - original_value * m
                dec_var_cost = func(var, constant_angles, T, w, dT, bodies, numSails, dv, NumSeg)

                # Restore original value
                var[i] = original_value

                # Update var and cost based on the calculated costs
                if cost > inc_var_cost:
                    var[i] = original_value + original_value * m
                    cost = inc_var_cost
                    no_change = 0
                elif cost > dec_var_cost:
                    var[i] = original_value - original_value * m
                    cost = dec_var_cost
                    no_change = 0
                else:
                    no_change += 1

                if printStatements == 1:
                    print("var #:", i, "m", m, "new cost:", cost)
                
                # Explicitly invoke garbage collection
                gc.collect()

    return var, cost
