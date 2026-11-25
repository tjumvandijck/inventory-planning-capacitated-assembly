import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def VisualizeCosts(stationary, p, save):
    # import data
    path = "C:\\Users\\u618593\\Desktop\\DynaplexPrivate\\IO_DynaPlex\\20261103 run output revision snellius 2\\"

    if stationary and p==50:
        case = "asml_assembly_nonstationary_NN10_p50_A_S_NoFE_"
        case_FE = "asml_assembly_nonstationary_NN10_p50_A_S_FE_"
    elif stationary and p==100:
        case = "asml_assembly_nonstationary_NN10_p100_A_S_NoFE4_"
        case_FE = "asml_assembly_nonstationary_NN10_p100_A_S_FE4_"
    elif not stationary and p==50:
        case = "asml_assembly_nonstationary_NN10_p50_A_NS_NoFE_"
        case_FE = "asml_assembly_nonstationary_NN10_p50_A_NS_FE_"
    elif not stationary and p==100:
        case = "asml_assembly_nonstationary_NN10_p100_A_NS_NoFE_"
        case_FE = "asml_assembly_nonstationary_NN10_p100_A_NS_FE_"

    if stationary:
        case_p100 = "asml_assembly_nonstationary_NN10_p100_A_S_NoFE_"
    else:
        case_p100 = "asml_assembly_nonstationary_NN10_p100_A_NS_NoFE_"


    # Read and convert data to numeric
    df_inv_costs_noFE = pd.read_csv(path + 'avg_inv_costs_per_policy_noFE_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_NoFE_lowN.csv', header=None)
    df_backlog_costs_noFE = pd.read_csv(path + 'avg_backlog_costs_per_policy_noFE_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_NoFE_lowN.csv', header=None)
    df_inv_costs_FE = pd.read_csv(path +  'avg_inv_costs_per_policy_FE_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_FE_lowN.csv', header=None)
    df_backlog_costs_FE = pd.read_csv(path +  'avg_backlog_costs_per_policy_FE_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_FE_lowN.csv', header=None, dtype=float)

    df_inv_costs_100 = pd.read_csv(path + 'avg_inv_costs_per_policy_FE_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_FE_lowN.csv', header=None)

    # Convert to lists and ensure numeric types
    df_inv_costs_noFE = pd.to_numeric(df_inv_costs_noFE.iloc[0, :], errors='coerce').dropna().tolist()
    df_backlog_costs_noFE = pd.to_numeric(df_backlog_costs_noFE.iloc[0,:], errors='coerce').dropna().tolist()

    df_inv_costs_FE = pd.to_numeric(df_inv_costs_FE.iloc[0,:], errors='coerce').dropna().tolist()
    df_backlog_costs_FE = pd.to_numeric(df_backlog_costs_FE.iloc[0,:], errors='coerce').dropna().tolist()

    df_inv_costs_100 = pd.to_numeric(df_inv_costs_100.iloc[0,:], errors='coerce').dropna().tolist()


    # Check if the first 6 elements are similar
    if df_inv_costs_noFE[:6] == df_inv_costs_FE[:6]:
        # Add the last element of df_inv_costs_FE to df_inv_costs_noFE
        df_inv_costs_noFE.append(df_inv_costs_FE[-1])

    # Check if the first 6 elements are similar
    if df_backlog_costs_noFE[:6] == df_backlog_costs_FE[:6]:
        # Add the last element to the list
        df_backlog_costs_noFE.append(df_backlog_costs_FE[-1])

    #
    # policies = ['BS-N', 'BS-N-S', 'BS-R', 'BS-R-S', 'BS-CB', 'BS-CB-S', 'DCL-A', 'DCL-A-FE']
    # policy_order = [2, 5, 1, 4, 0, 3, 6, 7]

    # Set the right order of the policies
    policies = ['BS-N',  'BS-R', 'BS-TW', 'BS-N-S', 'BS-R-S', 'BS-TW-S', 'DCL-A', 'DCL-A-FE']
    policy_order = [0, 2, 4, 1, 3, 5, 6, 7]

    # policies = ['BS-N',  'BS-R', 'BS-TW', 'BS-N-S', 'BS-R-S', 'BS-TW-S']
    # policy_order = [2, 1, 0, 5, 4, 3]

    # policies = ['BS-N', 'BS-R', 'BS-TW']
    # policy_order = [2, 1, 0]

    # Set the right order of the costs
    df_inv_costs_total = [df_inv_costs_noFE[i] for i in policy_order]
    df_backlog_costs_total = [df_backlog_costs_noFE[i] for i in policy_order]

    # Number of policies
    num_policies = len(policies)
    num_bars_per_stack = 2 # Inventory and backlog costs
    bar_width = 0.35
    bar_spacing = 0.05  # Space between bars
    x = np.arange(num_policies)

    # Create the plot
    # fig, ax = plt.subplots(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.rcParams.update({'font.size': 16})
    # Plot the bars
    bars1 = ax.bar(x - bar_width/2 - bar_spacing/2, df_inv_costs_total, bar_width, label='Inventory costs', alpha=0.7)
    bars2 = ax.bar(x + bar_width/2 + bar_spacing/2, df_backlog_costs_total, bar_width, label='Backlog costs', alpha=0.7)

    max_cost = max(max(df_inv_costs_total), max(df_backlog_costs_total))

    # Add text annotations
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.008*max_cost, f'{height:.0f}', ha='center', va='bottom', fontsize=12)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.008*max_cost, f'{height:.0f}', ha='center', va='bottom', fontsize=12)


    # Customize the plot
    ax.set_ylabel('Average cost per trajectory', fontsize = 14)
    # ax.set_title('Costs breakdown per Policy')
    ax.tick_params(axis='y', labelsize=14)  # Increase font size of the y-axis tick labels

    ax.set_xticks(x)
    ax.set_xticklabels([f'{policies[i]}' for i in range(num_policies)], fontsize = 14)

    # Adjust y-axis limits to add space for the legend
    max_cost = max(max(df_inv_costs_total), max(df_backlog_costs_total))
    min_cost = min(min(df_inv_costs_total), min(df_backlog_costs_total))

    ax.set_ylim(0, max_cost * 1.3)

    # Add legend
    plt.legend(fontsize=14, loc = 'upper left')
    if save:
        if stationary and p==50:
            plt.savefig("NN10_S_A_p50_withvalues.pdf", bbox_inches='tight')
        elif stationary and p==100:
            plt.savefig("NN10_S_A_p100_withvalues.pdf", bbox_inches='tight')
        elif not stationary and p==50:
            plt.savefig("NN10_NS_A_p50_withvalues.pdf", bbox_inches='tight')
        elif not stationary and p==100:
            plt.savefig("cost_graphs.pdf", bbox_inches='tight')

    plt.show()



if __name__ == "__main__":
    # Example usage:
    stationary = False
    p = 100
    save = True

    VisualizeCosts(stationary, p, save)


# # Create the plot
# fig, ax1 = plt.subplots(figsize=(10, 5))
#
# # Plot the inventory costs on the left y-axis
# bars1 = ax1.bar(x - bar_width/2 - bar_spacing/2, df_inv_costs_total, bar_width, label='Inventory costs', alpha=0.7, color='b')
# ax1.set_ylabel('Avg. inventory cost per trajectory', color='b')
# ax1.tick_params(axis='y', labelcolor='b')
#
# # Calculate the y-axis limits for inventory costs
# min_inv_cost = min(df_inv_costs_total)
# max_inv_cost = max(df_inv_costs_total)
# ax1.set_ylim(min_inv_cost * 0.8, max_inv_cost * 1.2)
#
# # Add text annotations for inventory costs
# for bar in bars1:
#     height = bar.get_height()
#     ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.008 * max_inv_cost, f'{height:.0f}', ha='center', va='bottom', fontsize=6, color='b')
#
# # Create a second y-axis for backlog costs
# ax2 = ax1.twinx()
#
# # Plot the backlog costs on the right y-axis
# bars2 = ax2.bar(x + bar_width/2 + bar_spacing/2, df_backlog_costs_total, bar_width, label='Backlog costs', alpha=0.7, color='r')
# ax2.set_ylabel('Avg. backlog cost per trajectory', color='r')
# ax2.tick_params(axis='y', labelcolor='r')
#
# # Calculate the y-axis limits for backlog costs
# min_backlog_cost = min(df_backlog_costs_total)
# max_backlog_cost = max(df_backlog_costs_total)
# ax2.set_ylim(min_backlog_cost * 0.8, max_backlog_cost * 1.2)
#
# # Add text annotations for backlog costs
# for bar in bars2:
#     height = bar.get_height()
#     ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.008 * max_backlog_cost, f'{height:.0f}', ha='center', va='bottom', fontsize=6, color='r')
#
# # Customize the plot
# ax1.set_xticks(x)
# ax1.set_xticklabels([f'{policies[i]}' for i in range(num_policies)])
#
# # Add legend
# fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
#
# # plt.savefig("NN10_S_A_p50_withvalues_dual_yaxis.pdf", bbox_inches='tight')
# plt.show()