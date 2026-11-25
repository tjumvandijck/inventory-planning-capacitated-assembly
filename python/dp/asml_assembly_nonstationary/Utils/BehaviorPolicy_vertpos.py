import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data

def VisualizePolicyTimeDependent(stationary, p, save, all_policies, lowest_vertical_pos):
    path = "C:\\Users\\u618593\\Desktop\\DynaplexPrivate\\IO_DynaPlex\\20261103 run output revision snellius 2\\"
    if stationary and p==50:
        case = "asml_assembly_nonstationary_NN10_p50_A_S_NoFE_"
        case_FE = "asml_assembly_nonstationary_NN10_p50_A_S_FE_"
    elif stationary and p==100:
        case = "asml_assembly_nonstationary_NN10_p100_A_S_NoFE_"
        case_FE = "asml_assembly_nonstationary_NN10_p100_A_S_FE_"
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
    df_action_per_vertpos_per_node_FE = pd.read_csv(path + 'avg_production_per_node_per_vertpos_per_policy_FE_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_FE_lowN.csv', header=None).dropna(axis=1)
    df_action_per_vertpos_per_node_noFE_original = pd.read_csv(path + 'avg_production_per_node_per_vertpos_per_policy_noFE_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_NoFE_lowN.csv', header=None).dropna(axis=1)
    df_action_per_vertpos_per_node_noFE = pd.read_csv(path + 'avg_production_per_node_per_vertpos_per_policy_noFE_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_NoFE_lowN.csv', header=None).dropna(axis=1)

    # # Identify the rows to be added from df_action_per_vertpos_per_node_FE
    rows_to_add = df_action_per_vertpos_per_node_FE.iloc[6::7]
    #
    # # Insert these rows into the appropriate positions in df_action_per_vertpos_per_node_noFE
    count = 0
    policies = ['BS-N', 'BS-N-S', 'BS-R', 'BS-R-S', 'BS-TW', 'BS-TW-S']
    num_policies = len(policies)
    for i, row in rows_to_add.iterrows():
        insert_position = count * 8 + 6
        df_action_per_vertpos_per_node_noFE = pd.concat([df_action_per_vertpos_per_node_noFE.iloc[:insert_position], row.to_frame().T, df_action_per_vertpos_per_node_noFE.iloc[insert_position:]], ignore_index=True)
        count += 1

    original_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


    if all_policies:
        # policies = ['BS-N', 'BS-N-S', 'BS-R', 'BS-R-S', 'BS-TW', 'BS-TW-S', 'DCL-A', 'DCL-A-FE']
        # # Reorder each set of 8 rows according to policy_order
        # policy_order = [2, 5, 1, 4, 0, 3, 6, 7]
        #
        policies = ['BS-N', 'BS-N-S', 'BS-R', 'BS-R-S', 'BS-TW', 'BS-TW-S']
        # Reorder each set of 8 rows according to policy_order
        policy_order = [2,5,1,4,0,3]
        color_indices = [0,1,2,3,4,5,6,7]
        num_policies = len(policies)
        policy_order2 = [x + num_policies for x in policy_order]
        policy_order3 = [x + 2*num_policies for x in policy_order]
        policy_order4 = [x + 3*num_policies for x in policy_order]
        policy_order = {0: policy_order, 1: policy_order2, 2: policy_order3, 3: policy_order4}
        chunks = {}
        for i in range(4):
            chunks[i] = df_action_per_vertpos_per_node_noFE.iloc[i*num_policies:(i+1)*num_policies].reindex(policy_order[i])
        df_action_per_vertpos_per_node_noFE = pd.concat(chunks.values(), ignore_index=True)
        df_avg_action_per_vertpos_per_policy = df_action_per_vertpos_per_node_noFE.mean(axis=1)

        df_transformed = pd.DataFrame()
        for i in range(num_policies):
            df_transformed = pd.concat([df_transformed, df_avg_action_per_vertpos_per_policy.iloc[i::num_policies].reset_index(drop=True)], axis=1)
        df_transformed = df_transformed.T.reset_index(drop=True)
    else:
        policies = ['BS-N-S', 'BS-R-S','BS-TW-S', 'DCL-A', 'DCL-A-FE']
        color_indices = [0,2,4,6,7]
        # Reorder each set of 8 rows according to policy_order
        policy_order = [1,3, 5, 6, 7]
        policy_order2 = [x + 8 for x in policy_order]
        policy_order3 = [x + 16 for x in policy_order]
        policy_order4 = [x + 24 for x in policy_order]
        policy_order = {0: policy_order, 1: policy_order2, 2: policy_order3, 3: policy_order4}
        chunks = {}
        for i in range(4):
            chunks[i] = df_action_per_vertpos_per_node_noFE.iloc[i * 8:(i + 1) * 8].reindex(policy_order[i])
        df_action_per_vertpos_per_node_noFE = pd.concat(chunks.values(), ignore_index=True)
        df_avg_action_per_vertpos_per_policy = df_action_per_vertpos_per_node_noFE.mean(axis=1)

        df_transformed = pd.DataFrame()
        for i in range(len(policies)):
            df_transformed = pd.concat(
                [df_transformed, df_avg_action_per_vertpos_per_policy.iloc[i::len(policies)].reset_index(drop=True)], axis=1)
        df_transformed = df_transformed.T.reset_index(drop=True)
    # Set the right order of the nodes
    # node_order = [1, 0, 3, 2, 4, 5, 6, 7, 8, 9]
    node_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df_action_per_vertpos_per_node_noFE = df_action_per_vertpos_per_node_noFE[node_order]


    # The dataframe now contains only the policies selected above: five rows
    # per vertical position in the focused view, or six in the all-policy view.
    num_policies = len(policies)
    if (lowest_vertical_pos):
        for i in range(4):
            policies_to_print_index = [(i*num_policies) + x for x in range(len(policies))]
            df_filtered = df_action_per_vertpos_per_node_noFE.iloc[policies_to_print_index]


            # Bar chart characteristics

            if all_policies:
                bar_width = 0.35
                bar_spacing = 0.08  # Space between bars in a pair
                pair_spacing = 0.5  # Space between pairs
            else:
                bar_width = 0.55
                bar_spacing = 0.15  # Space between bars in a pair
                pair_spacing = 0.6  # Space between pairs
            num_policies = df_filtered.shape[0]
            num_nodes = df_filtered.shape[1]


            # Create x positions for the bars
            x = np.arange(num_nodes) * (num_policies * (bar_width + bar_spacing) + pair_spacing)

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 6))

            max_cost = df_filtered.max().max()
            min_cost = df_filtered.min().min()


            # Plot the bars
            for j in range(num_policies):
                bars = ax.bar(x + j * (bar_width + bar_spacing), df_filtered.iloc[j, :], bar_width, label=f'{policies[j]}', color =original_colors[color_indices[j]], alpha=0.7)
                # Add text annotations
                # for bar in bars:
                #     height = bar.get_height()
                #     ax.text(bar.get_x() + bar.get_width() / 2, height + 0.008*max_cost, f'{height:.3f}', ha='center', va='bottom', fontsize=6)

            # Customize the plot
            ax.set_ylabel('Average production quantity', fontsize = 12)
            ax.tick_params(axis='y', labelsize=12)  # Increase font size of the y-axis tick labels
            ax.set_xticks(x + (num_policies - 1) * (bar_width + bar_spacing) / 2)
            ax.set_xticklabels([f'Node {num_nodes - j}' for j in range(num_nodes)], fontsize = 12)

            # Adjust y-axis limits to add space for the legend
            # Include zero-valued node output instead of cropping it below the
            # visible range (in particular, this can occur for Node 1).
            ax.set_ylim(0.9*min_cost, max_cost * 1.05)

            # Place the policy legend horizontally above the plotting area.
            fig.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, 0.91),
                ncol=num_policies,
                fontsize=14,
                frameon=False,
                columnspacing=1.5,
                handlelength=1.5,
            )
            fig.subplots_adjust(top=0.84)

            if save:
                if all_policies:
                    plt.savefig(f'NN10_NS_A_p100_allpolicies_VerticalPos{i}'+".pdf", bbox_inches='tight')
                else:
                    plt.savefig(f'NN10_NS_A_p100_focus_VerticalPos{i}' + ".pdf", bbox_inches='tight')

            plt.show()
    else:
        # Bar chart characteristics
        num_policies = len(policies)
        vertical_pos = [7,8,9,10]
        num_vert_pos = len(vertical_pos)

        if all_policies:
            bar_width = 0.5
            bar_spacing = 0.1  # Space between bars in a pair
            pair_spacing = 0.5  # Space between pairs
        else:
            bar_width = 0.85
            bar_spacing = 0.15  # Space between bars in a pair
            pair_spacing = 0.9  # Space between pairs

        # Adjust y-axis limits to add space for the legend
        max_cost = df_transformed.max().max()
        min_cost = df_transformed.min().min()

        # Create x positions for the bars
        x = np.arange(num_vert_pos) * (num_policies * (bar_width + bar_spacing) + pair_spacing)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.rcParams.update({'font.size': 12})

        # Plot the bars for each group
        for i in range(num_policies):
            bars = ax.bar(x + i * (bar_width + bar_spacing), df_transformed.iloc[i, :], bar_width, label=f'{policies[i]}', color =original_colors[color_indices[i]], alpha=0.7)
            # for bar in bars:
            #     height = bar.get_height()
            #     ax.text(bar.get_x() + bar.get_width() / 2, height + 0.008*max_cost, f'{height:.3f}', ha='center', va='bottom', fontsize=5.5)

        # Customize the plot
        ax.set_ylabel('Average production quantity', fontsize = 14)
        ax.tick_params(axis='y', labelsize=14)  # Increase font size of the y-axis tick labels
        ax.set_xticks(x + (num_policies - 1) * (bar_width + bar_spacing) / 2)
        ax.set_xticklabels([f'Demand level {vertical_pos[i]}' for i in range(num_vert_pos)], fontsize = 14)

        ax.set_ylim(min_cost*0.9, max_cost * 1.05)
        # Place the policy legend horizontally above the plotting area.
        fig.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 0.91),
            ncol=num_policies,
            fontsize=14,
            frameon=False,
            columnspacing=1.5,
            handlelength=1.5,
        )
        fig.subplots_adjust(top=0.84)
        if save:
            if stationary and p==50:
                plt.savefig("NN10_S_A_p50_avg_prod_quantity_per_vertpos.pdf", bbox_inches='tight')
            elif stationary and p==100:
                plt.savefig("NN10_S_A_p100_avg_prod_quantity_per_vertpos.pdf", bbox_inches='tight')
            elif not stationary and p==50:
                if all_policies:
                    plt.savefig("NN10_NS_A_p50_avg_prod_quantity_per_vertpos_all_policies.pdf", bbox_inches='tight')
                else:
                    plt.savefig("NN10_NS_A_p50_avg_prod_quantity_per_vertpos_focus.pdf", bbox_inches='tight')
            elif not stationary and p==100:
                plt.savefig("NN10_NS_A_p100__avg_prod_quantity_per_vertpos.pdf", bbox_inches='tight')

        plt.show()



if __name__ == "__main__":
    # Example usage:
    stationary = False
    p = 100
    save = True
    all_policies = False
    lowest_vertical_pos = True
    print(save)
    VisualizePolicyTimeDependent(stationary, p, save, all_policies, lowest_vertical_pos)

