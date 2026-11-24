import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy



def objectiveP1_2(p1, changingPoints, returnObjective):
    p_indexs = [0, 1, 2]
    percentile = [0.9, 0.5, 0.1]
    time_slots = list(range(0, 18 + 1))
    probabilities_df = pd.DataFrame(index=[0, 1], columns=time_slots)
    probabilities_df.loc[:, :] = 0

    penalty = [1,2,1]
    probs = [[1-p1, p1]]
    means = [0]
    for time_slot in time_slots:
        if time_slot == 0:
            probabilities_df.loc[0, time_slot] = 1
            continue
        count_mean = 0
        for mean in means:
            count = 0
            for prob in probs[count_mean]:
                new_mean = mean + count
                probabilities_df.loc[new_mean, time_slot] += prob * probabilities_df.loc[mean, time_slot - 1]
                count += 1
            count_mean += 1
    cdf = probabilities_df.cumsum(axis=0)

    axis = 0
    objective_test = 0
    for p_index in p_indexs:
        for time_slot in time_slots:
            if (cdf.loc[axis][time_slot] < percentile[p_index] and time_slot < changingPoints[axis][p_index]):
                objective_test += (percentile[p_index] - cdf.loc[axis][time_slot])*penalty[p_index]
            elif (cdf.loc[axis][time_slot] > percentile[p_index] and time_slot >= changingPoints[axis][p_index]):
                objective_test += (cdf.loc[axis][time_slot] - percentile[p_index])*penalty[p_index]

    if returnObjective:
        return objective_test
    else:
        return objective_test, probabilities_df

def objectiveP2(p2, changingPoints, probabilities, axis, p1, returnObjective):
    probabilities_df = copy.deepcopy(probabilities)
    p_indexs = [0,1,2]
    percentile = [0.9,0.5,0.1]
    time_slots = list(range(0, 18 + 1))
    penalty = [1,2,1]

    # first prob could be zero, as that is already computed in probabilities_axis
    probs = [[0, p1], [1-p2, p2]]
    means = [axis - 1,axis]
    for time_slot in time_slots:
        if time_slot == 0:
            continue
        count_mean = 0
        for mean in means:
            count = 0
            for prob in probs[count_mean]:
                new_mean = mean + count
                probabilities_df.loc[new_mean, time_slot] += prob * probabilities_df.loc[mean, time_slot - 1]
                count += 1
            count_mean += 1
    cdf = probabilities_df.cumsum(axis=0)

    objective_test = 0
    for p_index in p_indexs:
        for time_slot in time_slots:
            if (cdf.loc[axis][time_slot] < percentile[p_index] and time_slot < changingPoints[axis][p_index]):
                objective_test += (percentile[p_index] - cdf.loc[axis][time_slot])*penalty[p_index]
            elif (cdf.loc[axis][time_slot] > percentile[p_index] and time_slot >= changingPoints[axis][p_index]):
                objective_test += (cdf.loc[axis][time_slot] - percentile[p_index])*penalty[p_index]


    if returnObjective:
        return objective_test
    else:
        return objective_test, probabilities_df

def greedyHeuristicP(changingPoints, horizon):
    # Keep track of the probabilities:
    time_slots = list(range(0, horizon + 1))
    probabilities = pd.DataFrame(index=[0,1,2,3], columns=time_slots)
    probabilities.loc[:, :] = 0
    total_objective = 0

    ######## Compute p1 ########
    ## Check feasibility
    # For now assume that changingPoints[0][0] = 1
    p_lowerbound1 = 1 - 0.9**(1/changingPoints[0][0])
    p_upperbound1 = 1 - 0.5**(1/(changingPoints[0][1] - 1))
    p_lowerbound2 = 1 - 0.5**(1/(changingPoints[0][1]))
    p_upperbound2 = 1 - 0.1**(1/(changingPoints[0][2] - 1))
    p_lowerbound3 = 1 - 0.1**(1/(changingPoints[0][2]))
    feasible = True
    if (p_lowerbound1 > p_upperbound1 or p_lowerbound1 > p_upperbound2 or p_lowerbound2 > p_upperbound1 or p_lowerbound2 > p_upperbound2 or p_lowerbound3 > p_upperbound1 or p_lowerbound3 > p_upperbound2):
        feasible = False

    if (feasible):
        # take average of max lower bound and min upper bound
        p1 = round(0.5*(max(p_lowerbound1, p_lowerbound2, p_lowerbound3) + min(p_upperbound1, p_upperbound2)),3)
    else:
        # here we have to compute the loss functions per p
        objective = {}
        for p in np.arange(round(min(p_lowerbound1, p_lowerbound2, p_lowerbound3), 2), round(max(p_upperbound1, p_upperbound2),2), 0.002):
            # objective[p] = objectiveP1(p,changingPoints)
            objective[p] = objectiveP1_2(p,changingPoints, True)
        p1 = round(min(objective, key=objective.get), 3)
    objective1, probabilities1 = objectiveP1_2(p1, changingPoints, False)
    total_objective += objective1

    # Compute the probabilities to be in a specific point on the second vertical axis, but just moving upwards
    probabilities_up_axis1 = [0] # this is the probability that at index - 1 you were at axis = 0, and you moved up at time period = index
    probabilities_axis0 = [1] # is the probability that at index t you are at axis = 0
    for t in range(1,horizon+1):
        probabilities_up_axis1.append((1-p1)**(t-1)*p1)
        probabilities_axis0.append((1-p1)**t)

    probabilities.loc[0] = probabilities_axis0
    ######## Compute p2 ########
    objectivep2 = {}
    for p in np.arange(0.01, 0.4, 0.001):
        if round(p,3) == 0.137:
            debug = 1
        objectivep2[round(p,3)] = objectiveP2(p,changingPoints, probabilities, 1, p1, True)
        if objectivep2[round(p,3)] == 0:
            break
    p2 = round(min(objectivep2, key=objectivep2.get), 3)
    objective2, probabilities2 = objectiveP2(p2, changingPoints, probabilities, 1, p1, False)
    probabilities.loc[1] = list(probabilities2.loc[1])
    total_objective += objective2

    ####### Compute p3 ########
    probabilities_points_axis1 = [0] # is the probability that at index t you are at axis = 1
    probabilities_up_axis2 = [0,0] # this is the probability that at index - 1 you were at axis = 1, and you moved up at time period = index
    for t in range(1,horizon+1):
        prob = sum(probabilities_up_axis1[t1] * (1 - p2) ** (t - t1) for t1 in range(t+1))
        probabilities_points_axis1.append(prob)
    for t in range(2,horizon+1):
        probabilities_up_axis2.append(probabilities_points_axis1[t-1]*p2)

    ######## Compute p3 ########
    objectivep3 = {}
    for p in np.arange(0.01, 0.4, 0.0002):
        objectivep3[p] = objectiveP2(p,changingPoints, probabilities, 2, p2, True)
        if objectivep3[p] == 0:
            break
    p3 = round(min(objectivep3, key=objectivep3.get), 3)
    objective3, probabilities3 = objectiveP2(p3, changingPoints, probabilities, 2, p2, False)
    probabilities.loc[2] = list(probabilities3.loc[2])
    total_objective += objective3

    return [p1, p2, p3], total_objective


### To test, but can be discarded ####
def printNormalDistr(base_prob, low_prob, high_prob):
    
    stdev_low = (low_prob - base_prob) / -1.282
    stdev_high = (high_prob - base_prob) / 1.282
    N = 100000
    
    
    print("Low: ", low_prob, "Stdev: ", stdev_low)
    
    samples_low = np.random.normal(loc=base_prob, scale=stdev_low, size=int(N))
    samples_high = np.random.normal(loc=base_prob, scale=stdev_high, size=int(N))
    samples = samples_low/2 + samples_high/2
    # samples = np.concatenate((samples_low, samples_high))
    
    # Sort the samples
    sorted_samples = np.sort(samples)

    # Compute the cumulative probability for each sample
    cdf = np.arange(1, N+1) / N
    
    
    print("Mean: ", np.mean(samples))
    
    
    p = 1
    
    # Compute the 10th percentile
    percentile_10th = np.percentile(samples, 10)
    print(f"10th Percentile: {percentile_10th:.4f}")
    
    
    # Compute the 10th percentile
    percentile_90th = np.percentile(samples, 90)
    print(f"90th Percentile: {percentile_90th:.4f}")
    
    # Plot a histogram of the samples
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=30, color='skyblue', edgecolor='black')
    plt.title('Histogram of Normally Distributed Samples')
    plt.xlabel('Sample Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    
    return 0



def objectiveP2_old(p2, changingPoints, probabilities_up_axis, probabilities_axis, axis, probabilities_axis0, p1, returnObjective):
    objective = 0
    timings = [-3,-2,-1,0,1,2,3]
    p_indexs = [0,1,2]
    percentile = [0.9,0.5,0.1]
    time_slots = list(range(0, 18 + 1))

    # probabilities = [[0 for _ in range(19)] for _ in range(3)]
    # probabilities[0] = probabilities_axis

    probs = [[0, p1], [1-p2, p2]]
    means = [0,1]
    means_df = [0,1,2,4]
    probabilities = pd.DataFrame(index=means_df, columns=time_slots)
    probabilities.loc[:, :] = 0
    probabilities.loc[0] = probabilities_axis
    for time_slot in time_slots:
        if time_slot == 0:
            continue
        count = 0
        for mean in means:
            count = 0
            for prob in probs[mean]:
                new_mean = mean + count
                probabilities.loc[new_mean, time_slot] += prob * probabilities.loc[mean, time_slot - 1]
                count += 1
    cdf = probabilities.cumsum(axis=0)

    objective_test = 0
    for p_index in p_indexs:
        for time_slot in time_slots:
            if (cdf.loc[1][time_slot] < percentile[p_index] and time_slot < changingPoints[axis][p_index]):
                objective_test += percentile[p_index] - cdf.loc[1][time_slot]
            elif (cdf.loc[1][time_slot] > percentile[p_index] and time_slot >= changingPoints[axis][p_index]):
                objective_test += cdf.loc[1][time_slot] - percentile[p_index]

    # for p_index in p_indexs:
    #     for timing in timings:
    #         # if 0 < changingPoints[axis][p_index] < 19 and 1 < changingPoints[axis][p_index] + timing < 19:
    #         if 1 < changingPoints[axis][p_index] + timing < 19:
    #             # last plus is to add all probabilities of lower vertical positions
    #             prob = sum(probabilities_up_axis[t]*(1-p2)**(changingPoints[axis][p_index] + timing - t) for t in range(changingPoints[axis][p_index] + timing + 1))  + probabilities_axis[changingPoints[axis][p_index] + timing] + probabilities_axis0[changingPoints[axis][p_index] + timing]
    #             if (prob < percentile[p_index] and timing < 0):
    #                 objective += percentile[p_index] - prob
    #             elif (prob > percentile[p_index] and timing >= 0):
    #                 objective += prob - percentile[p_index]


    if returnObjective:
        return objective_test
    else:
        return probabilities


def objectiveP1_old(p1, changingPoints):
    objective = 0

    # check if first jump satisfied, otherwise, add error to objective
    if (1-p1)**(changingPoints[0][0]-1) < 0.9 and changingPoints[0][0]-1 > 0:
        objective += 0.9 - (1 - p1)**(changingPoints[0][0]-1)
    if (1-p1)**changingPoints[0][0] > 0.9:
        objective += (1-p1)**changingPoints[0][0] - 0.9
    if (1-p1)**(changingPoints[0][0]+1) > 0.9:
        objective += (1-p1)**(changingPoints[0][0]+1) - 0.9

    # check second jumping point
    if (1-p1)**(changingPoints[0][1]-2) < 0.5:
        objective += 0.5 - (1-p1)**(changingPoints[0][1]-2)
        ### big error
        objective += 0.5
    if (1-p1)**(changingPoints[0][1]-1) < 0.5:
        objective += 0.5 - (1-p1)**(changingPoints[0][1]-1)
    if (1-p1)**(changingPoints[0][1]) > 0.5:
        objective +=  (1-p1)**(changingPoints[0][1]) - 0.5
    if (1-p1)**(changingPoints[0][1]+1) > 0.5:
        objective += (1-p1)**(changingPoints[0][1]+1) - 0.5
    if (1-p1)**(changingPoints[0][1]+2) > 0.5:
        objective += (1-p1)**(changingPoints[0][1]+2) - 0.5
        ## Big error
        objective += 0.5

    # check third jumping point
 # check second jumping point
    if (1-p1)**(changingPoints[0][2]-2) < 0.1:
        objective += 0.1 - (1-p1)**(changingPoints[0][2]-2)
        ### big error
        objective += 0.5
    if (1-p1)**(changingPoints[0][2]-1) < 0.1:
        objective += 0.1 - (1-p1)**(changingPoints[0][2]-1)
    if (1-p1)**(changingPoints[0][2]) > 0.1:
        objective +=  (1-p1)**(changingPoints[0][2]) - 0.1
    if (1-p1)**(changingPoints[0][2]+1) > 0.1:
        objective += (1-p1)**(changingPoints[0][2]+1) - 0.1
    if (1-p1)**(changingPoints[0][2]+2) > 0.1 and changingPoints[0][2]+2 < 19:
        objective += (1-p1)**(changingPoints[0][2]+2) - 0.1
        ## Big error
        objective += 0.5

    return objective
