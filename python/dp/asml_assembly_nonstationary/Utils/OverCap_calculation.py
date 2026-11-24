import pandas as pd
import numpy as np

def Distribution_general(initial_mean, max_mean, probs, horizon, initial_state, intial_move, minimum_pos):
    time_slots = list(range(0, horizon + 1))
    means = list(range(minimum_pos, max_mean + 1))
    probabilities = pd.DataFrame(index=means, columns=time_slots)
    probabilities.loc[:, :] = 0

    for time_slot in time_slots:
        if time_slot == 0:
            probabilities.loc[initial_mean, time_slot] = 1
        else:
            for mean in means:
                count = 0
                for prob in probs[mean - minimum_pos]:
                    new_mean = mean + intial_move + count
                    if new_mean < minimum_pos:
                        new_mean = minimum_pos
                    if new_mean > max_mean:
                        new_mean = max_mean
                    probabilities.loc[new_mean, time_slot] += prob * probabilities.loc[mean, time_slot - 1]
                    count += 1   
    
        
    # Compute the CDF per column
    cdf = probabilities.cumsum(axis=0)
    
    # Normalize the CDF to ensure the last value in each column is 1
    cdf = cdf.div(cdf.iloc[-1, :], axis=1)
    return probabilities, cdf

def Distribution(initial_mean, max_mean, prob_up, horizon):
    time_slots = list(range(0, horizon + 1))
    means = list(range(initial_mean, max_mean + 1))

    for prob in prob_up:
        probabilities = pd.DataFrame(index=means, columns=time_slots)
        probabilities.loc[:, :] = 0

        for time_slot in time_slots:
            if time_slot == 0:
                probabilities.loc[initial_mean, time_slot] = 1
            else:
                for mean in means:
                    if mean == initial_mean:
                        probabilities.loc[mean, time_slot] = (1 - prob) * probabilities.loc[mean, time_slot - 1]
                    elif mean == max_mean:
                        probabilities.loc[mean, time_slot] = probabilities.loc[mean, time_slot - 1] + \
                                                             prob * probabilities.loc[mean - 1, time_slot - 1]
                    else:
                        probabilities.loc[mean, time_slot] = (1 - prob) * probabilities.loc[mean, time_slot - 1] + \
                                                             prob * probabilities.loc[mean - 1, time_slot - 1]
    
    # Compute the CDF per column
    cdf = probabilities.cumsum(axis=0)
    
    # Normalize the CDF to ensure the last value in each column is 1
    cdf = cdf.div(cdf.iloc[-1, :], axis=1)
    return probabilities, cdf



# Compute the over capacity for the distribution given by the probabilities
def overCap_calc(probabilities, capacity, prob_up, horizon):
    time_slots = list(range(0, horizon + 1))

    overcap_endhorizon = []
    overcap_avg = []


    expected_demand = pd.Series(index=time_slots)
    overcap_timeslot = pd.Series(index=time_slots)
    for time_slot in time_slots:
        expected_demand.loc[time_slot] = np.dot(probabilities.loc[:, time_slot], probabilities.index)
        overcap_timeslot.loc[time_slot] = (capacity - expected_demand[time_slot]) / expected_demand[time_slot]

    overcap_endhorizon.append((capacity - expected_demand.loc[horizon]) / expected_demand.loc[horizon])
    overcap_avg.append(overcap_timeslot[1::].mean())

    return overcap_endhorizon, overcap_avg, expected_demand


def percentile_linear_interpolation(data, probs, target_percentile):
    """
    Compute the value corresponding to a given percentile using linear interpolation.

    Args:
    - data: A list or numpy array of sorted data points.
    - probabilities: A list or numpy array of cumulative probabilities corresponding to the data points.
    - target_percentile: The target percentile to compute (e.g., 0.10 for the 10th percentile).

    Returns:
    - Interpolated value corresponding to the target percentile.
    """
    # Ensure data and probabilities are numpy arrays for easy manipulation
    data = np.array(data)
    probs = np.array(probs)
    
    if probs[0] > target_percentile:
        return data[0]
    else:
        # Find the two probabilities that bound the target_percentile, this should be cumulative
        for i in range(len(probs) - 1):
            if probs[i] <= target_percentile <= probs[i + 1]:
                # Perform linear interpolation
                x_i = data[i]
                x_i1 = data[i + 1]
                p_i = probs[i]
                p_i1 = probs[i + 1]
                
                # Interpolated value
                result = x_i + ((target_percentile - p_i) / (p_i1 - p_i)) * (x_i1 - x_i)
                return result

    # If the target_percentile is outside the bounds of the probabilities, return None or handle as needed
    return data[-1]


def low_base_high_input(changingPoints, horizon, initial_mean, vertical_pos):
    time_slots = list(range(0, horizon + 1))
    low_prob = np.full(horizon + 1, initial_mean)
    base_prob = np.full(horizon + 1, initial_mean)
    high_prob = np.full(horizon + 1, initial_mean)
    prob = [high_prob, base_prob, low_prob]
    for time_slot in time_slots:
        for k in range(len(vertical_pos)-1):
        # 3 is fixed for high, base, low
            for x in range(3):
                if time_slot == changingPoints[k][x]:
                    prob[x][time_slot:] += 1

    return prob
def low_base_high(probabilities, horizon, initial_mean):
    time_slots = list(range(0, horizon + 1))
    low_prob = np.full(horizon + 1, initial_mean)
    base_prob = np.full(horizon + 1, initial_mean)
    high_prob = np.full(horizon + 1, initial_mean)
    data = probabilities.index.tolist()
    
    for time_slot in time_slots:
        # if time_slot == 0:
            # low_prob[time_slot] += initial_mean
            # base_prob[time_slot] += initial_mean
            # high_prob[time_slot] += initial_mean
            
        # else:
            # low_prob[time_slot] += percentile_linear_interpolation(data, probabilities[time_slot], 0.1)
            # base_prob[time_slot] += percentile_linear_interpolation(data, probabilities[time_slot], 0.5)
            # high_prob[time_slot] += percentile_linear_interpolation(data, probabilities[time_slot], 0.9)
        
        
        # Find the two probabilities that bound the target_percentile, this should be cumulative
        for i in range(len(probabilities[time_slot])):
            if probabilities[time_slot][data[i]] >= 0.1:
                low_prob[time_slot] += i
                break
        for i in range(len(probabilities[time_slot])):
            if probabilities[time_slot][data[i]] >= 0.5:
                base_prob[time_slot] += i
                break
        for i in range(len(probabilities[time_slot])):
            if probabilities[time_slot][data[i]] >= 0.9:
                high_prob[time_slot] += i
                break
            
        
    return low_prob, base_prob, high_prob