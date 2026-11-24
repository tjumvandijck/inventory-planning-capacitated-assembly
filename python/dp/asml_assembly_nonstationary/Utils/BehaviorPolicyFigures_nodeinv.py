import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap

# import data


def VisualizePolicyNodeInv(stationary, p, save, all_policies):
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
        case = "avg_inv_per_node_per_policy_noFE_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_NoFE_lowN"
        case_FE = "avg_inv_per_node_per_policy_FE_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_FE_lowN"

    if stationary:
        case_p100 = "avg_inv_per_node_per_policy_FE_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_FE_lowN"
    else:
        case_p100 = "avg_inv_per_node_per_policy_noFE_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_NoFE_lowN"

    # Read and convert data to numeric
    df_inv_per_node_noFE = pd.read_csv(path + case + '.csv', header=None).dropna(axis=1)
    df_inv_per_nodes_FE = pd.read_csv(path + case_FE + '.csv', header=None).dropna(axis=1)

    df_inv_per_nodes_100 = pd.read_csv(path + case_p100 + '.csv', header=None).dropna(axis=1)


    # Check if the first 6 rows are equal
    if df_inv_per_node_noFE.iloc[:6].equals(df_inv_per_nodes_FE.iloc[:6]):
        # Append the last row of df_inv_per_nodes_FE to df_inv_per_node_noFE
        df_inv_per_node_noFE = pd.concat([df_inv_per_node_noFE, df_inv_per_nodes_FE.iloc[[-1]]], ignore_index=True)


    original_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Set the right order of the policies
    if all_policies:
        # policies = ['BS-N', 'BS-N-S', 'BS-R', 'BS-R-S', 'BS-TW', 'BS-TW-S', 'DCL-A', 'DCL-A-FE']
        # policy_order = [2, 5, 1, 4, 0, 3, 6, 7]
        # df_inv_per_node_total = df_inv_per_node_noFE.reindex(policy_order)

        policies = ['BS-N', 'BS-N-S', 'BS-R', 'BS-R-S', 'BS-TW', 'BS-TW-S']
        policy_order = [2, 5, 1, 4, 0, 3]
        df_inv_per_node_total = df_inv_per_nodes_FE.reindex(policy_order)
        color_indices = [0,1,2,3,4,5,6,7]

    else:
        policies = ['BS-N-S', 'BS-R-S', 'BS-TW-S', 'DCL-A', 'DCL-A-FE']
        policy_order = [1,3,5,6,7]
        df_inv_per_node_total = df_inv_per_node_noFE.reindex(policy_order)
        color_indices = [0,2,4,6,7]

        # policies = ['BS-N', 'BS-TW', 'DCL-A-FE']
        # policy_order = [5,3,7]
        # df_inv_per_node_total = df_inv_per_node_noFE.reindex(policy_order)
        # color_indices = [0,1,2]


    # if stationary:
    #     node_order = [1, 0, 3, 2, 4, 5, 6, 7, 8, 9]
    #     df_inv_per_node_total = df_inv_per_node_total[node_order]
    #     nodes_print = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    #
    # else:
    node_order = [0, 1, 3, 5, 2, 4, 6, 7, 8, 9]
    df_inv_per_node_total = df_inv_per_node_total[node_order]
    nodes_print = [10, 9, 7, 5, 8, 6, 4, 3, 2, 1]



    # Bar chart characteristics
    if all_policies:
        num_policies = len(policies)

        num_nodes = df_inv_per_node_total.shape[1]
        bar_width = 0.35
        bar_spacing = 0.08  # Space between bars in a pair
        pair_spacing = 0.5  # Space between pairs
    else:
        # Bar chart characteristics
        num_policies = len(policies)
        num_nodes = df_inv_per_node_total.shape[1]
        bar_width = 0.7
        bar_spacing = 0.2  # Space between bars in a pair
        pair_spacing = 0.8  # Space between pairs

    # Create x positions for the bars
    x = np.arange(num_nodes) * (num_policies * (bar_width + bar_spacing) + pair_spacing)

    # Create the plot
    fig, ax = plt.subplots(figsize=(11, 6))
    plt.rcParams.update({'font.size': 14})

    # Plot the bars
    for i in range(num_policies):
        # color = cmap(0.3 + 0.6 * i / num_policies)  # Generate a color from the colormap
        ax.bar(x + i * (bar_width + bar_spacing), df_inv_per_node_total.iloc[i, :], bar_width, label=f'{policies[i]}', color =original_colors[color_indices[i]], alpha=0.7)

    # Customize the plot
    ax.set_ylabel('Average on-hand inventory level', fontsize = 14)
    ax.tick_params(axis='y', labelsize=14)  # Increase font size of the y-axis tick labels
    # ax.set_title('Inventory levels comparison per node for all policies')
    ax.set_xticks(x + (num_policies - 1) * (bar_width + bar_spacing) / 2)
    ax.set_xticklabels([f'Node {nodes_print[i]}' for i in range(num_nodes)], fontsize = 12)
    ax.set_ylim(bottom=0)


    x_start = x[1] + - bar_spacing
    x_end = x[3] + num_policies * (bar_width) + bar_spacing

    # These y coordinates are relative to the x-axis (0 = x-axis), so the
    # inventory scale can remain at zero while annotations use whitespace below.
    bracket_y = -0.10
    text_y = -0.12
    bracket_height = 0.025
    addition_text = 0.2 if stationary else 0.31
    xaxis_transform = ax.get_xaxis_transform()

    ax.plot([x_start, x_end], [bracket_y, bracket_y], color='gray', lw=2,
            clip_on=False, transform=xaxis_transform)
    ax.plot([x_start, x_start - addition_text],
            [bracket_y, bracket_y + bracket_height], color='gray', lw=2,
            clip_on=False, transform=xaxis_transform)
    ax.plot([x_end, x_end + addition_text],
            [bracket_y, bracket_y + bracket_height], color='gray', lw=2,
            clip_on=False, transform=xaxis_transform)

    # Add text under the bracket
    ax.text((x_start + x_end) / 2, text_y, "Pred. of node 2",
            ha='center', va='top', fontsize=12, clip_on=False,
            transform=xaxis_transform)

    x_start = x[4] - bar_spacing
    x_end = x[5] + num_policies * (bar_width) + bar_spacing


    ax.plot([x_start, x_end], [bracket_y, bracket_y], color='gray', lw=2,
            clip_on=False, transform=xaxis_transform)
    ax.plot([x_start, x_start - addition_text],
            [bracket_y, bracket_y + bracket_height], color='gray', lw=2,
            clip_on=False, transform=xaxis_transform)
    ax.plot([x_end, x_end + addition_text],
            [bracket_y, bracket_y + bracket_height], color='gray', lw=2,
            clip_on=False, transform=xaxis_transform)

    # Add text under the bracket
    ax.text((x_start + x_end) / 2, text_y, "Pred. of node 4",
            ha='center', va='top', fontsize=12, clip_on=False,
            transform=xaxis_transform)

    # Adjust y-axis limits to add space for the legend
    # max_cost = df_inv_per_nodes_100.max().max()
    # ax.set_ylim(0, 1.1*max_cost)

    # if not stationary:
        # Adjust the layout to add space for the text
    if stationary:
        plt.subplots_adjust(bottom=0.25)  # Space for annotations below x-axis
    else:
        plt.subplots_adjust(bottom=0.25)

    # Add legend
    # if not stationary and p==50 and not all_policies:
    plt.legend(fontsize=12, title='Policies', title_fontsize='12', loc = 'upper right')

    # else:
        # plt.legend(fontsize=11, title='Policies', title_fontsize='11', bbox_to_anchor=(1.15,1))

    if save:
        if stationary and p==50:
            plt.savefig("MSOM-NN10_S_A_p50_avg_node_inv.pdf", bbox_inches='tight')
        elif stationary and p==100:
            plt.savefig("MSOM-NN10_S_A_p100_avg_node_inv.pdf", bbox_inches='tight')
        elif not stationary and p==50:
            if all_policies:
                plt.savefig("MSOM-NN10_NS_A_p50_avg_node_inv_allpolicies.pdf", bbox_inches='tight')
            else:
                plt.savefig("MSOM-Poster-nodeinv.pdf", bbox_inches='tight')
        elif not stationary and p==100:
            plt.savefig("NN10_NS_A_p100_avg_node_inv.pdf", bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Example usage:
    stationary = False
    p = 100
    save = True
    all_policies = False

    VisualizePolicyNodeInv(stationary, p, save, all_policies)
