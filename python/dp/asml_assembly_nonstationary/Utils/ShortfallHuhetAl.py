import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import csv
import math
import time

def ComputeShortfall_infinite(demand_prob, demand_values, capacity, minimum_order, vertical_pos):
    # Define parameters
    d_values = np.array(demand_values)  # Possible values of d
    d_probabilities = np.array(demand_prob)  # Probabilities for each value in d_values
    c = capacity  # Capacity
    initial_v = 0  # Starting value of v

    # Simulation parameters
    num_trajectories = 2000  # Number of independent simulations
    num_steps = 500  # Number of time steps for each trajectory

    # Storage for the results
    empirical_probs_str_vec = []
    # Run simulations
    for t in vertical_pos:
        v_counts = np.zeros(500)
        for _ in range(num_trajectories):
            v = initial_v
            for _ in range(num_steps):
                d = t + np.random.choice(d_values, p=d_probabilities)
                v = max(v + d - c, 0)
                if v > 100:
                    debug = 1
                else:
                    v_counts[v] += 1

        # Normalize counts to get empirical probabilities
        total_counts = sum(v_counts)
        empirical_probs = v_counts/total_counts
        empirical_probs_str = ";".join(map(str, empirical_probs))
        empirical_probs_str_vec.append(empirical_probs_str)

    # Plotting the empirical distribution function (EDF)
    # plt.step(empirical_probs, list(range(0,100)), where='post')
    # plt.xlabel("Value of v")
    # plt.ylabel("Cumulative Probability")
    # plt.title("Empirical Distribution Function (EDF) of v")
    # plt.grid()
    # plt.show()

    print(empirical_probs_str_vec)

    # # Write the vector to a CSV file
    # with open("empirical_probs2.csv", mode="w", newline="") as file:
    #     writer = csv.writer(file)
    #     for row in empirical_probs_str_vec:
    #         writer.writerow([row])

    return empirical_probs

def ComputeShortfall_trajectory(noise_prob, noise_values, capacity, minimum_order, vertical_pos, discount_factor, max_tree, start_mean_dem, start_noise, move_prob, lead_times, average_shortfall = False):
    # Define parameters
    d_values = np.array(demand_values)  # Possible values of d
    d_probabilities = np.array(demand_prob)  # Probabilities for each value in d_values
    c = capacity  # Capacity
    initial_v = 0  # Starting value of v
    # Simulation parameters
    num_trajectories = 20000  # Number of independent simulations

    noiseLength = len(noise_values)

    # Create the event lists
    demands = []
    for i in range(max_tree - start_mean_dem + 1):
        # Create the first list
        list1 = [discount_factor * move_prob[i][x] * noise_prob[y] for x in range(len(move_prob[i])) for y in range(len(noise_prob))]

        # Create the second list
        list2 = [(1-discount_factor)* move_prob[i][x] * noise_prob[y] for x in range(len(move_prob[i])) for y in range(len(noise_prob))]

        # Concatenate the two lists
        result = list1 + list2
        demands.append(result)

    indices = [i for i in range(len(demands[0]))]

    # Storage for the results
    empirical_probs_str_vec = []
    count_ver_pos = [0 for _ in range(len(vertical_pos))]

    start_time = time.time()
    v_counts = np.zeros((len(vertical_pos), 100))
    # Run simulations
    for _ in range(num_trajectories):
        v = initial_v
        run = True
        vert_pos = 0
        while run:
            event = np.random.choice(indices, p=demands[vert_pos])
            finalstate = math.floor(event/ (len(demands[vert_pos]) / 2))
            if finalstate == 1:
                run = False
                bern = math.floor((event - (len(demands[vert_pos]) / 2)) / noiseLength)
            else:
                bern = math.floor(event / noiseLength)

            if vert_pos != max_tree - start_mean_dem:
                vert_pos += bern

            demand = start_mean_dem + vert_pos + (event % noiseLength + start_noise)
            v = max(v + demand - c, 0)
            v_counts[vert_pos][v] += 1
            count_ver_pos[vert_pos] += 1

    empirical_probs_vec = []
    # Normalize counts to get empirical probabilities
    for row in range(len(v_counts)):
        empirical_probs = [v_counts[row][x] / count_ver_pos[row] for x in range(len(v_counts[row]))]
        empirical_probs_str = ";".join(map(str, empirical_probs))

        # print(empirical_probs_str)
        empirical_probs_str_vec.append(empirical_probs_str)
        empirical_probs_vec.append(empirical_probs)

    end_time = time.time()
    time_shortfall_method = end_time - start_time
    print(f"Time for shortfall {time_shortfall_method}")
    a = 1
    # # New shortfall method to compute shortfall per echelon lead time
    # average_vertical_pos = [[] for _ in range(len(vertical_pos))]
    # if average_shortfall:
    #     for v in range(len(vertical_pos)):
    #         initial_pos = [0 for _ in range(len(vertical_pos))]
    #         initial_pos[v] += 1
    #         avg_vert_pos = []
    #         for lt in range(0,lead_times[0]):
    #             if lt == 0:
    #                 avg_vert_pos_vec = initial_pos
    #             else:
    #                 avg_vert_pos_vec = avg_vert_pos[lt-1]
    #             new_avg_vert_pos = [0 for _ in range(len(vertical_pos))]
    #             for j in range(len(avg_vert_pos_vec)):
    #                 for k in range(len(move_prob[i])):
    #                     if j + k < len(vertical_pos):
    #                         new_avg_vert_pos[j+k] += avg_vert_pos_vec[j] * move_prob[j][k]
    #                     else:
    #                         new_avg_vert_pos[-1] += avg_vert_pos_vec[j] * move_prob[j][k]
    #             avg_vert_pos.append(new_avg_vert_pos)
    #             if lt+1 in lead_times:
    #                 # Compute the average vertical position over the echelon lead times
    #                 avg_vert_pos_lt = [sum(avg_vert_pos[v][j] for v in range(len(avg_vert_pos)))/len(avg_vert_pos) for j in range(len(vertical_pos))]
    #                 average_vertical_pos[v].append(avg_vert_pos_lt)

    #     shortfall_avg = []
    #     for i in range(len(vertical_pos)):
    #         for j in range(len(lead_times)):
    #             # Average shortfall per vertical position per lead time
    #             avg_shortfall = [sum(average_vertical_pos[i][j][k]*empirical_probs_vec[k][l] for k in range(len(vertical_pos))) for l in range(len(empirical_probs_vec[0]))]
    #             shortfall_avg.append(avg_shortfall)

    # print("Average Shortfall over echelon lead times:")
    # for row in range(len(shortfall_avg)):
    #     empirical_probs_str = ";".join(map(str, shortfall_avg[row]))
    #     print(empirical_probs_str)
    return empirical_probs_vec

# NS=H OLD
demand_prob = [0.08, 0.15, 0.2, 0.23, 0.15, 0.08, 0.05, 0.03, 0.02, 0.01]
demand_values = list(range(-3, -3 + len(demand_prob)))
minimum_order = 4
vertical_pos = [7,8,9,10]
capacity = [10,11]
move_prob = [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2], [0.8,0.2]]
vertical_pos = [7,8,9,10]
capacity = [10]
discount_factor = 0.9444444
max_tree = 10
start_mean_dem = 7
start_noise = -3
lead_tiems = [3,2]


# NS - L
# demand_prob = [0.03, 0.1148, 0.7104, 0.1248, 0.01, 0.01]
# # demand_prob = [ 0.03, 0.1148, 0.7104, 0.1248, 0.01, 0.01 ]
demand_prob = [0.16, 0.23, 0.26, 0.2, 0.1, 0.05]
demand_values = [-2, -1, 0, 1, 2, 3]
minimum_order = 0
vertical_pos = [2,3,4]
capacity = [5]
discount_factor = 0.9444444
max_tree = 4
start_mean_dem = 2
start_noise = -2
move_prob = [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]
lead_times = [3]


# Largest case: NS-H
# demand_prob = [0.04, 0.18, 0.21, 0.23, 0.14, 0.09, 0.07, 0.04]
# demand_values = list(range(-3, -3 + len(demand_prob)))
# minimum_order = 4
# vertical_pos = [7,8,9,10]
# capacity = [10,11]
# move_prob = [[0.803, 0.197], [0.805, 0.195], [0.8300, 0.1700], [1, 0]]
# vertical_pos = [7,8,9,10]
# capacity = [10]
# discount_factor = 0.9444444
# max_tree = 10
# start_mean_dem = 7
# start_noise = -3
# lead_times = [6]

# S - L
# demand_prob = [0.04, 0.12, 0.16, 0.2, 0.23, 0.14, 0.06, 0.03, 0.02]
# demand_values = [-2, -1, 0, 1, 2, 3, 4, 5, 6]
# minimum_order = 0
# vertical_pos = [2,3,4]
# capacity = [5]
# discount_factor = 0.9444444
# max_tree = 2
# start_mean_dem = 2
# start_noise = -2
# move_prob = [[1.0, 0.0]]
# lead_times = [3]

# # S - H
# demand_prob = [0.02, 0.05, 0.1, 0.14, 0.20, 0.19, 0.1, 0.06, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01]
# demand_values = [-3 + x for x in range(len(demand_prob))]
# minimum_order = 0
# vertical_pos = [2,3,4]
# capacity = [11]
# discount_factor = 0.9444444
# max_tree = 7
# start_mean_dem = 7
# start_noise = -3
# move_prob = [[1.0, 0.0]]
# lead_times = [3]




# Small case - same CV as largest case
# demand_prob = [0.03, 0.1048, 0.7404, 0.0948, 0.02, 0.01]
# # demand_prob = [ 0.03, 0.1148, 0.7104, 0.1248, 0.01, 0.01 ]
# # demand_prob = [0.16, 0.23, 0.26, 0.2, 0.1, 0.05]
# demand_values = [-2, -1, 0, 1, 2, 3]
# minimum_order = 0
# vertical_pos = [2,3,4]
# capacity = [4]
# discount_factor = 0.9444444
# max_tree = 4
# start_mean_dem = 2
# start_noise = -2
# move_prob = [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]


ComputeShortfall_trajectory(demand_prob, demand_values, capacity[0], minimum_order, vertical_pos, discount_factor, max_tree, start_mean_dem, start_noise, move_prob, lead_times, True)
