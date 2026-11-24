from Utils.OverCap_calculation import overCap_calc, Distribution, low_base_high, Distribution_general
from Utils.Corridor_Estimation import greedyHeuristicP
# from Utils.ShortfallHuhetAl import ComputeShortfall
import numpy as np
import matplotlib.pyplot as plt

demand_prob = [0.08, 0.15, 0.2, 0.23, 0.15, 0.08, 0.05, 0.03, 0.02, 0.01]
demand_values = list(range(-3, -3 + len(demand_prob)))
minimum_order = 4
vertical_pos = [7,8,9,10]
capacity = [10]


# ComputeShortfall(demand_prob, demand_values, capacity[0], minimum_order, vertical_pos)

horizon = 18

changingPoints = [[1,6,12],[4,10,19],[8,16,19]]
p = greedyHeuristicP(changingPoints, horizon)

# base_prob = 9.31380714
# low_prob = 8.00534537
# high_prob = 9.86276143
# target_percentiles = [low_prob, base_prob, high_prob]


# Parameters
initial_mean = 7
max_mean = 10
prob_up = [0.2]
horizon = 18
capacity = 10
# probs = [[0,0.5,0.5],[0.1,0.7,0.2],[0.1,0.8,0.2],[0.2,0.6,0.2],[0.3,0.7,0]]
probs_fitted = [[1-p[0], p[0]], [1-p[1], p[1]], [1-p[2], p[2]], [1-p[2], p[2]], ]

initial_move = 0
minimum_pos = 7


probabilities_gen, cdf_gen = Distribution_general(initial_mean, max_mean, probs_fitted, horizon, initial_mean, initial_move, minimum_pos)
overcap_endhorizon, overcap_avg, expected_demand = overCap_calc(probabilities_gen, capacity, prob_up, horizon)
low_prob, base_prob, high_prob = low_base_high(cdf_gen, horizon, initial_mean)
print("Average overcapacity over the planning horizon: " + str(overcap_avg))
print("Overcapacity at the end of the planning horizon: " + str(overcap_endhorizon))

low_prob_int = np.round(low_prob)
base_prob_int = np.round(base_prob)
high_prob_int = np.round(high_prob)

# Plotting low_prob, base_prob, and high_prob in one graph
## MAINUTILS FOR FINAL FITTING PLOT
time_slots = np.arange(horizon + 1)

plt.figure(figsize=(10, 6))
plt.plot(time_slots, low_prob, label='Low Probability', marker='o')
plt.plot(time_slots, base_prob, label='Base Probability', marker='o')
plt.plot(time_slots, high_prob, label='High Probability', marker='o')

plt.plot(time_slots, low_prob_int, label='Low Probability int', marker='x')
plt.plot(time_slots, base_prob_int, label='Base Probability int', marker='x')
plt.plot(time_slots, high_prob_int, label='High Probability int', marker='x')


plt.xlabel('Time Slots')
plt.ylabel('Probability')
plt.title('Probabilities Over Time')
plt.xticks(time_slots)  # Set x-ticks to be the time slots from 0 to 18
plt.grid(True)
plt.legend()
# plt.savefig("DemandCorridor_WO_interpolation")
plt.show()


a = 1