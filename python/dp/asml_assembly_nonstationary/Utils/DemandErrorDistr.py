import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def plotForecastErrorDistr(probabilites, xticks):
    # Create the bar plot
    plt.figure(figsize=(8, 5))
    plt.bar(xticks, probabilites, color='#1f77b4', edgecolor='black', width=0.6)
    plt.rcParams.update({'font.size': 12})

    # Add labels and title
    plt.ylim(0, 1.1 * max(probabilites))  # Set y-axis limit
    plt.xlabel('Forecast error', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.xticks(xticks)  # Custom x-axis labels

    # Show grid for better readability
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()
    plt.savefig("ForecastErrorDistrNS-leadingcase.pdf")

    plt.show()


noiseProbSmallNS = [0.16, 0.23, 0.26, 0.2, 0.1, 0.05]
xticksSmallNS = [-2, -1, 0, 1, 2, 3]
noiseProblargeNS = [0.08, 0.15, 0.2, 0.23, 0.15, 0.08, 0.05, 0.03, 0.02, 0.01]
xticksLargeNS = [-3 + x for x in range(len(noiseProblargeNS))]
noiseProbSmallS = [0.04, 0.12, 0.16, 0.2, 0.23, 0.14, 0.06, 0.03, 0.02]
xticksSmallS = [0+x for x in range(len(noiseProbSmallS))]
noiseProblargeS = [0.02, 0.05, 0.1, 0.14, 0.2, 0.19, 0.1, 0.06, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01]
xticksLargeS = [4+x for x in range(len(noiseProblargeS))]


noiseProblargeNS = [0.04, 0.18, 0.21, 0.23, 0.14, 0.09, 0.07, 0.04]
xticksLeadingCaseNS = [- 3 + x for x in range(len(noiseProblargeNS))]

plotForecastErrorDistr(noiseProblargeNS, xticksLargeNS)
