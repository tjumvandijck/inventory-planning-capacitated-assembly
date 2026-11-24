from OverCap_calculation import overCap_calc, Distribution, low_base_high, Distribution_general, low_base_high_input
from Corridor_Estimation import greedyHeuristicP
# from ShortfallHuhetAl import ComputeShortfall
import numpy as np
import matplotlib.pyplot as plt

# demand_prob = [0.08, 0.15, 0.2, 0.23, 0.15, 0.08, 0.05, 0.03, 0.02, 0.01]
# demand_values = list(range(-3, -3 + len(demand_prob)))
# minimum_order = 4
vertical_pos = [7,8,9,10]
capacity = [10,11]


# ComputeShortfall(demand_prob, demand_values, capacity, minimum_order, vertical_pos)

horizon = 18

# Fully fitted cone
changingPoints = [[1,4,11],[3,9,19],[7,14,19]]

# To test
# changingPoints = [[2,6,14],[5,12,19],[10,17,19]]


time_slots = np.arange(19)
y_values = np.zeros((3, len(time_slots))) + 7

# Update y-values based on changing points
for i, points in enumerate(changingPoints):
    for j, point in enumerate(points):
        if point < len(time_slots):
            y_values[j, point:] = i + 1 + 7



p, objective = greedyHeuristicP(changingPoints, horizon)
print(p)
print(objective)

base_prob = 9.31380714
low_prob = 8.00534537
high_prob = 9.86276143
target_percentiles = [low_prob, base_prob, high_prob]

# Parameters
initial_mean = 7
max_mean = 10
prob_up = [0.2]
horizon = 18
capacity = 10
# probs = [[0,0.5,0.5],[0.1,0.7,0.2],[0.1,0.8,0.2],[0.2,0.6,0.2],[0.3,0.7,0]]
probs_fitted = [[1-p[0], p[0]], [1-p[1], p[1]], [1-p[2], p[2]], [1-p[2], p[2]], ]
probs_fitted_input = [[0.803,0.197], [0.805,0.195], [0.8300,0.1700], [1,0] ]

initial_move = 0
minimum_pos = 7


probabilities_gen, cdf_gen = Distribution_general(initial_mean, max_mean, probs_fitted, horizon, initial_mean, initial_move, minimum_pos)
overcap_endhorizon, overcap_avg, expected_demand = overCap_calc(probabilities_gen, capacity, prob_up, horizon)
low_prob, base_prob, high_prob = low_base_high(cdf_gen, horizon, initial_mean)


prob = low_base_high_input(changingPoints, horizon, initial_mean, vertical_pos)
high_prob_input = prob[0]
base_prob_input = prob[1]
low_prob_input = prob[2]

# probabilities_gen_input, cdf_gen_input = Distribution_general(initial_mean, max_mean, probs_fitted_input, horizon, initial_mean, initial_move, minimum_pos)
# low_prob_input, base_prob_input, high_prob_input = low_base_high(cdf_gen_input, horizon, initial_mean)

low_prob_int = np.round(low_prob)
base_prob_int = np.round(base_prob)
high_prob_int = np.round(high_prob)

# Plotting low_prob, base_prob, and high_prob in one graph
time_slots = np.arange(horizon + 1)

# Plotting the lines and markers
plt.figure(figsize=(10, 6))
line_low, = plt.plot(time_slots, low_prob_input, color='grey', dashes = (5,5))
line_base, = plt.plot(time_slots, base_prob_input, color='black')
line_high, = plt.plot(time_slots, high_prob_input, color='grey', linestyle='--')

marker_colors_low = ['orange' if low_prob_input[i] != low_prob[i] else 'grey' for i in range(len(time_slots))]
marker_colors_base = ['red' if base_prob_input[i] != base_prob[i] else 'black' for i in range(len(time_slots))]
marker_colors_high = ['tomato' if high_prob_input[i] != high_prob[i] else 'grey' for i in range(len(time_slots))]


marker_low = plt.scatter(time_slots, low_prob, marker='x', color=marker_colors_low)
marker_base = plt.scatter(time_slots, base_prob, marker='x', color=marker_colors_base)
marker_high = plt.scatter(time_slots, high_prob, marker='x', color=marker_colors_high)

# Creating custom legend entries
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='grey', linestyle='--', label='High: Input'),
    Line2D([0], [0], marker='x', color='grey', linestyle='None', label='High: Estimate'),
    Line2D([0], [0], color='black', label='Base: Input'),
    Line2D([0], [0], marker='x', color='black', linestyle='None', label='Base: Estimate'),
    Line2D([0], [0], color='grey', dashes = (5,5), label='Low: Input'),
    Line2D([0], [0], marker='x', color='grey', linestyle='None', label='Low: Estimate')
]

# Adding the legend to the plot
plt.legend(handles=legend_elements)

plt.xlabel('Time Slots')
plt.ylabel('Average Demand')
plt.xticks(time_slots)  # Set x-ticks to be the time slots from 0 to 18
plt.grid(False)
# plt.savefig("DemandConeFitted.pdf", bbox_inches='tight')
plt.show()

a = 1