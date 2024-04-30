import h5py
import os
import shutil
import click
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import itertools
from pandas.plotting import table
import plotly.graph_objects as go
import metrics
colormap = plt.cm.jet

def closest_bin(query, bin_array):
    for i in range(len(bin_array)-1):
        if query >= bin_array[i] and query < bin_array[i+1]:
            if (bin_array[i+1] - query) < (query - bin_array[i]):
                return bin_array[i+1]
            else:
                return bin_array[i]
    return bin_array[-1]

def construct_result_dictionary(save_path, lprot_test, ax_load_test, boundary_test, rc_gt_test, rc_pred_final_test, reserve_capacity_bins):
    #Compute mape on the test results
    mape_test, mape_std_test, avg_percentage_err_test = metrics.mape(rc_gt_test, rc_pred_final_test)
    mape_test_fw, mape_std_test_fw, avg_percentage_err_test_fw = metrics.mape(rc_gt_test[:,0], rc_pred_final_test[:,0])
    mape_test_bw, mape_std_test_bw, avg_percentage_err_test_bw = metrics.mape(rc_gt_test[:,1], rc_pred_final_test[:,1])

    #Create the result dictionary
    scenarios = itertools.product(np.unique(lprot_test), np.unique(ax_load_test), np.unique(boundary_test))
    scenario_keys = [' '.join(sc) for sc in scenarios]
    error_dict = { 'error': {'best_config':{'rc':{'mean': -1, 'std': -1},
                                             'rc+':{'mean': -1, 'std': -1, 'inst_err': [],
                                                    'scenario': {sc: {'mean': -1, 'std': -1, 'gt':[], 'gt_categ': [], 'pred':[], 'inst_err': []} for sc in scenario_keys}},
                                             'rc-':{'mean': -1, 'std': -1, 'inst_err': [],
                                                    'scenario': {sc: {'mean': -1, 'std': -1, 'gt':[], 'gt_categ': [], 'pred':[], 'inst_err': []} for sc in scenario_keys}}
                    }}}

    #Instance and scenario based error
    for inst in range(rc_gt_test.shape[0]):
        scen_name = ' '.join((lprot_test[inst], ax_load_test[inst], boundary_test[inst]))
        if scen_name not in error_dict['error']['best_config']['rc+']['scenario'].keys():
            error_dict['error']['best_config']['rc+']['scenario'][scen_name] = {'mean': -1, 'std': -1, 'gt':[], 'gt_categ': [], 'pred':[], 'inst_err': []}
        if scen_name not in error_dict['error']['best_config']['rc-']['scenario'].keys():
            error_dict['error']['best_config']['rc-']['scenario'][scen_name] = {'mean': -1, 'std': -1, 'gt':[], 'gt_categ': [], 'pred':[], 'inst_err': []}
        fw_rc_bin = closest_bin(rc_gt_test[inst][0]*100, reserve_capacity_bins)
        bw_rc_bin = closest_bin(rc_gt_test[inst][1]*100, reserve_capacity_bins)
        error_dict['error']['best_config']['rc+']['scenario'][scen_name]['inst_err'].append(avg_percentage_err_test_fw[inst])
        error_dict['error']['best_config']['rc+']['scenario'][scen_name]['gt'].append(rc_gt_test[inst][0])
        error_dict['error']['best_config']['rc+']['scenario'][scen_name]['gt_categ'].append(fw_rc_bin)
        error_dict['error']['best_config']['rc+']['scenario'][scen_name]['pred'].append(rc_pred_final_test[inst][0])
        error_dict['error']['best_config']['rc+']['inst_err'].append(avg_percentage_err_test_fw[inst])
        error_dict['error']['best_config']['rc-']['scenario'][scen_name]['inst_err'].append(avg_percentage_err_test_bw[inst])
        error_dict['error']['best_config']['rc-']['scenario'][scen_name]['gt'].append(rc_gt_test[inst][1])
        error_dict['error']['best_config']['rc-']['scenario'][scen_name]['gt_categ'].append(bw_rc_bin)
        error_dict['error']['best_config']['rc-']['scenario'][scen_name]['pred'].append(rc_pred_final_test[inst][1])
        error_dict['error']['best_config']['rc-']['inst_err'].append(avg_percentage_err_test_bw[inst])
    for scen in error_dict['error']['best_config']['rc+']['scenario'].keys():
        all_inst = np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['inst_err'])
        error_dict['error']['best_config']['rc+']['scenario'][scen]['mean'] = np.mean(all_inst)
        error_dict['error']['best_config']['rc+']['scenario'][scen]['std'] = np.std(all_inst)
    for scen in error_dict['error']['best_config']['rc-']['scenario'].keys():
        all_inst = np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['inst_err'])
        error_dict['error']['best_config']['rc-']['scenario'][scen]['mean'] = np.mean(all_inst)
        error_dict['error']['best_config']['rc-']['scenario'][scen]['std'] = np.std(all_inst)
    #Overall error
    error_dict['error']['best_config']['rc']['mean'] = mape_test
    error_dict['error']['best_config']['rc']['std'] = mape_std_test
    error_dict['error']['best_config']['rc+']['mean'] = mape_test_fw
    error_dict['error']['best_config']['rc+']['std'] =  mape_std_test_fw
    error_dict['error']['best_config']['rc-']['mean'] = mape_test_bw
    error_dict['error']['best_config']['rc-']['std'] = mape_std_test_bw
    #Save the error dictionary
    np.save(os.path.join(save_path, 'error_dict.npy'), error_dict)

def plot_pred_vs_gt_combi(error_dict, save_dict):
    #cetagory=> 0:collapse-cons, monotonic, symm 1: axes yield ratio. 2: boundary condition
    # Create a dataframe object
    for scen in error_dict['error']['best_config']['rc-']['scenario'].keys():
        concat_x, concat_y, concat_hue, concat_type_rc = np.array([]), np.array([]), np.array([]), np.array([])
        nb_elem = np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['gt']).shape
        concat_x = np.concatenate((concat_x, np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['gt'], dtype=float)))
        concat_y = np.concatenate((concat_y, np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['pred'])))
        concat_hue = np.concatenate((concat_hue, np.array(scen).repeat(nb_elem)))
        concat_type_rc = np.concatenate((concat_type_rc, np.array('fw').repeat(nb_elem)))
        
        nb_elem = np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['gt']).shape
        concat_x = np.concatenate((concat_x, np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['gt'], dtype=float)))
        concat_y = np.concatenate((concat_y, np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['pred'])))
        concat_hue = np.concatenate((concat_hue, np.array(scen).repeat(nb_elem)))
        concat_type_rc = np.concatenate((concat_type_rc, np.array('bw').repeat(nb_elem)))

        sns_dict = {'x':concat_x,
                    'y':concat_y,
                    'scenario': concat_hue,
                    'fwbw': concat_type_rc}
        # Convert the dictionary to a DataFrame
        plt.figure(figsize=(22, 10))
        df = pd.DataFrame(sns_dict)
        row_values = df['fwbw'].unique()
        # Generate lmplot with trendline (ci=None)
        g = sns.lmplot(data=df, x="x", y="y", hue='scenario', palette="bright", row="fwbw", ci=None, height=8, aspect=1, legend=False, scatter=False)
        # Add scatter plot on top
        for row_val, ax_g in zip(row_values, g.axes.flat):
            # Filter DataFrame for the current row value
            subset_df = df[df['fwbw'] == row_val]
            sns.scatterplot(data=subset_df, x="x", y="y", hue='scenario', palette="pastel", s=12, alpha=0.7, legend=False, ax = ax_g)
        # Adjust legend if needed
        g.axes[0][0].legend(ncol=10, bbox_to_anchor=(0.0, 1.05), loc='upper left', fontsize="12", frameon=False)

        for ax in g.axes_dict.values():
            ax.axline((0.4, 0.4), slope=1, c=".2", ls="--", zorder=1)
        # Remove titles
        g.set_titles("")
        g.axes[1,0].set_xlabel('True Reserve Capacity', fontsize=16)
        g.axes[0,0].set_ylabel("Pred " + "$R_{M}(-)$", fontsize=18)
        g.axes[1,0].set_ylabel("Pred " + "$R_{M}(+)$", fontsize=18)
        g.axes[0,0].set_aspect('equal')
        g.axes[1,0].set_aspect('equal')
        reserve_capacity_bins_pred = np.arange(0.4, 1.05, 0.1)
        reserve_capacity_bins_gt = np.arange(0.4, 1.05, 0.1)
        g.axes[0,0].set_xticks(reserve_capacity_bins_gt)
        g.axes[0,0].set_xticklabels([f'{tick:.2f}' for tick in reserve_capacity_bins_gt],rotation=30)
        g.axes[1,0].set_xticklabels([f'{tick:.2f}' for tick in reserve_capacity_bins_gt],rotation=30, fontsize=14)
        g.axes[0,0].set_yticks(reserve_capacity_bins_pred)
        g.axes[0,0].set_yticklabels([f'{tick:.2f}' for tick in reserve_capacity_bins_pred], fontsize=14)
        g.axes[1,0].set_yticks(reserve_capacity_bins_pred)
        g.axes[1,0].set_yticklabels([f'{tick:.2f}' for tick in reserve_capacity_bins_pred], fontsize=14)
        # Add a big title to the entire plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        g.fig.suptitle('Ground Truth vs Predicted Reserve Capacity for ' + (save_dict.split('/')[-1]).upper(), fontsize=12)
        if not os.path.exists(os.path.join(save_dict, 'combinations')):
            os.makedirs(os.path.join(save_dict, 'combinations'), exist_ok=True)

        plt.savefig(os.path.join(save_dict, 'combinations', 'pred_vs_gt_combi_' + save_dict.split('/')[-1]+ '_scenario_' + (scen)+ '.png'))
        plt.close()

def plot_pred_vs_gt(error_dict, save_dict, category, threshold=0.6, lprot=['Symmetric', 'Collapse_consistent', 'Monotonic']):
    #cetagory=> 0:collapse-cons, monotonic, symm 1: axes yield ratio. 2: boundary condition
    concat_x, concat_y, concat_hue, concat_type_rc = np.array([]), np.array([]), np.array([]), np.array([])
    # Create a dataframe object
    for scen in error_dict['error']['best_config']['rc-']['scenario'].keys():
        lprot_current = scen.split(' ')[0]
        if lprot_current not in lprot:
            continue
        gt = np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['gt'])
        gt_valid = np.where(gt >= threshold)[0]
        nb_elem = np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['gt'])[gt_valid].shape
        concat_x = np.concatenate((concat_x, np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['gt'], dtype=float)[gt_valid]))
        concat_y = np.concatenate((concat_y, np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['pred'])[gt_valid]))
        concat_hue = np.concatenate((concat_hue, np.array(scen.split(' ')[category]).repeat(nb_elem)))
        concat_type_rc = np.concatenate((concat_type_rc, np.array('fw').repeat(nb_elem)))
    for scen in error_dict['error']['best_config']['rc+']['scenario'].keys():
        lprot_current = scen.split(' ')[0]
        if lprot_current not in lprot:
            continue
        gt = np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['gt'])
        gt_valid = np.where(gt >= threshold)[0]
        nb_elem = np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['gt'])[gt_valid].shape
        concat_x = np.concatenate((concat_x, np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['gt'], dtype=float)[gt_valid]))
        concat_y = np.concatenate((concat_y, np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['pred'])[gt_valid]))
        concat_hue = np.concatenate((concat_hue, np.array(scen.split(' ')[category]).repeat(nb_elem)))
        concat_type_rc = np.concatenate((concat_type_rc, np.array('bw').repeat(nb_elem)))

    sns_dict = {'x':concat_x,
                'y':concat_y,
                'scenario': concat_hue,
                'fwbw': concat_type_rc}
    # Convert the dictionary to a DataFrame
    plt.figure(figsize=(22, 6))
    df = pd.DataFrame(sns_dict)
    row_values = df['fwbw'].unique()
    # Generate lmplot with trendline (ci=None)
    #g = sns.lmplot(data=df, x="x", y="y", hue='scenario', palette="bright", row="fwbw", fit_reg=True, x_ci='sd', ci=50, n_boot=5, height=8, aspect=1, legend=False, scatter=False)
    g = sns.lmplot(data=df, x="x", y="y", hue='scenario', palette="bright", row="fwbw", ci=95, height=8, aspect=1, legend=False, scatter=False)

    # Add scatter plot on top
    for row_val, ax_g in zip(row_values, g.axes.flat):
        # Filter DataFrame for the current row value
        subset_df = df[df['fwbw'] == row_val]
        sns.scatterplot(data=subset_df, x="x", y="y", hue='scenario', palette="pastel", s=12, alpha=0.5, legend=False, ax = ax_g)
    # Adjust legend if needed
    g.axes[0][0].legend(ncol=10, bbox_to_anchor=(0.0, 1.05), loc='upper left', fontsize="12", frameon=False)

    for ax in g.axes_dict.values():
        ax.axline((threshold, threshold), slope=1, c=".2", ls="--", zorder=1)
    # Remove titles
    g.set_titles("")
    g.axes[1,0].set_xlabel('True Reserve Capacity', fontsize=16)
    g.axes[0,0].set_ylabel("Pred " + "$R_{M}(-)$", fontsize=18)
    g.axes[1,0].set_ylabel("Pred " + "$R_{M}(+)$", fontsize=18)
    g.axes[0,0].set_aspect('equal')
    g.axes[1,0].set_aspect('equal')
    reserve_capacity_bins_pred = np.arange(0.5, 1.05, 0.1)#np.arange(0.4, 1.05, 0.1)
    reserve_capacity_bins_gt = np.arange(threshold, 1.05, 0.1) #np.arange(0.4, 1.05, 0.1)
    g.axes[0,0].set_xticks(reserve_capacity_bins_gt)
    g.axes[0,0].set_xticklabels([f'{tick:.2f}' for tick in reserve_capacity_bins_gt],rotation=30)
    g.axes[1,0].set_xticklabels([f'{tick:.2f}' for tick in reserve_capacity_bins_gt],rotation=30, fontsize=14)
    g.axes[0,0].set_yticks(reserve_capacity_bins_pred)
    g.axes[0,0].set_yticklabels([f'{tick:.2f}' for tick in reserve_capacity_bins_pred], fontsize=14)
    g.axes[1,0].set_yticks(reserve_capacity_bins_pred)
    g.axes[1,0].set_yticklabels([f'{tick:.2f}' for tick in reserve_capacity_bins_pred], fontsize=14)
    # Add a big title to the entire plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    g.fig.suptitle('Ground Truth vs Predicted Reserve Capacity for ' + (save_dict.split('/')[-1]).upper(), fontsize=12)
    plt.savefig(os.path.join(save_dict, 'pred_vs_gt_' + save_dict.split('/')[-1]+ '_category_' + str(category)+ '_scen_' +  lprot[0] +'_ci.png'))

def plot_mape_vs_rc_combi(error_dict, save_dict):
    #cetagory=> 0:collapse-cons, monotonic, symm 1: axes yield ratio. 2: boundary condition
    # Create a dataframe object
    for scen in error_dict['error']['best_config']['rc-']['scenario'].keys():
        concat_x, concat_y, concat_hue, concat_type_rc = np.array([]), np.array([]), np.array([]), np.array([])
        nb_elem = np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['gt_categ']).shape
        concat_x = np.concatenate((concat_x, np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['gt_categ'], dtype=float)))
        concat_y = np.concatenate((concat_y, np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['inst_err'])))
        concat_hue = np.concatenate((concat_hue, np.array(scen).repeat(nb_elem)))
        concat_type_rc = np.concatenate((concat_type_rc, np.array('fw').repeat(nb_elem)))
        nb_elem = np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['gt_categ']).shape
        concat_x = np.concatenate((concat_x, np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['gt_categ'], dtype=float)))
        concat_y = np.concatenate((concat_y, np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['inst_err'])))
        concat_hue = np.concatenate((concat_hue, np.array(scen).repeat(nb_elem)))
        concat_type_rc = np.concatenate((concat_type_rc, np.array('bw').repeat(nb_elem)))
        sns_dict = {'x':concat_x,
                    'y':concat_y,
                    'scenario': concat_hue,
                    'fwbw': concat_type_rc}
        # Convert the dictionary to a DataFrame
        fig = plt.figure(figsize=(25, 10))
        df = pd.DataFrame(sns_dict)
        row_values = df['fwbw'].unique()
        g = sns.FacetGrid(df, row="fwbw", hue='scenario', height=10, aspect=2, )#gridspec_kws={"hspace":0.0})
        for row_val, ax_g in zip(row_values, g.axes.flat):
            # Filter DataFrame for the current row value
            subset_df = df[df['fwbw'] == row_val]
            # Plot stripplot and pointplot for the current row value
            sns.stripplot(x="x", y="y", hue='scenario', data=subset_df, jitter=True, palette="muted", native_scale=True, alpha=0.7, zorder=-1, legend=False, ax=ax_g)
            pointplot = sns.pointplot(x="x", y="y", hue='scenario', data=subset_df, errorbar=('ci', 95),  native_scale=True, palette="bright", markersize=6, markeredgewidth=2,  ax=ax_g)
            # Get handles and labels from the last pointplot
            handles, labels =pointplot.get_legend_handles_labels()
        handles, labels = g.axes[0,0].get_legend_handles_labels()
        g.axes[0,0].legend(loc='upper left', bbox_to_anchor=(0.0, 1.05), ncol=6, fontsize="20", frameon=False)
        g.axes[1,0].get_legend().set_visible(False) 
        #Remove titles
        g.set_titles("")
        g.axes[1,0].set_xlabel('True Reserve Capacity', fontsize=22)
        g.axes[0,0].set_ylabel("APE of " + "$R_{M}(-)$", fontsize=22)
        g.axes[1,0].set_ylabel("APE of " + "$R_{M}(+)$", fontsize=22)
        g.axes[0,0].set_aspect('equal')
        g.axes[1,0].set_aspect('equal')
        ace_bins = [1e0, 1e1, 1e2]
        reserve_capacity_bins_gt = np.arange(40, 105, 5)
        for ax_g in g.axes.flat:
            ax_g.set_yscale('log', base=10)  # Apply log scale
            ax_g.set_aspect('auto', adjustable='datalim')  # Manually set the aspect ratio
            ax_g.yaxis.set_major_locator(plt.FixedLocator([1e0, 1e1, 1e2]))  # Set your desired ticks
            ax_g.set_ylim(bottom=1e-0, top=1e2)

        g.axes[0,0].set_xticks(reserve_capacity_bins_gt)
        g.axes[0,0].set_xticklabels([f'{tick/100:.2f}' for tick in reserve_capacity_bins_gt],rotation=30)
        g.axes[1,0].set_xticklabels([f'{tick/100:.2f}' for tick in reserve_capacity_bins_gt],rotation=30, fontsize=20)
        g.axes[0,0].set_yticks(ace_bins)
        g.axes[0,0].set_yticklabels([f'{tick}' for tick in ace_bins], fontsize=20)
        g.axes[1,0].set_yticks(ace_bins)
        g.axes[1,0].set_yticklabels([f'{tick:.2f}' for tick in ace_bins], fontsize=20)
        # Add a big title to the entire plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        g.axes[0,0].tick_params(axis='x', pad=10)
        g.fig.suptitle('Average Percengage Error for ' + (save_dict.split('/')[-1]).upper(), fontsize=20)
        if not os.path.exists(os.path.join(save_dict, 'combinations')):
            os.makedirs(os.path.join(save_dict, 'combinations'), exist_ok=True)
        plt.savefig(os.path.join(save_dict, 'combinations', 'mape_vs_rc_combi' + save_dict.split('/')[-1]+ '_scenario_' + (scen)+ '.png'))
        plt.close()

def plot_mape_vs_rc(error_dict, save_dict, category):
    #cetagory=> 0:collapse-cons, monotonic, symm 1: axes yield ratio. 2: boundary condition
    concat_x, concat_y, concat_hue, concat_type_rc = np.array([]), np.array([]), np.array([]), np.array([])
    # Create a dataframe object
    for scen in error_dict['error']['best_config']['rc-']['scenario'].keys():
        nb_elem = np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['gt_categ']).shape
        concat_x = np.concatenate((concat_x, np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['gt_categ'], dtype=float)))
        concat_y = np.concatenate((concat_y, np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['inst_err'])))
        concat_hue = np.concatenate((concat_hue, np.array(scen.split(' ')[category]).repeat(nb_elem)))
        concat_type_rc = np.concatenate((concat_type_rc, np.array('fw').repeat(nb_elem)))
    for scen in error_dict['error']['best_config']['rc+']['scenario'].keys():
        nb_elem = np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['gt_categ']).shape
        concat_x = np.concatenate((concat_x, np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['gt_categ'], dtype=float)))
        concat_y = np.concatenate((concat_y, np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['inst_err'])))
        concat_hue = np.concatenate((concat_hue, np.array(scen.split(' ')[category]).repeat(nb_elem)))
        concat_type_rc = np.concatenate((concat_type_rc, np.array('bw').repeat(nb_elem)))
    sns_dict = {'x':concat_x,
                'y':concat_y,
                'scenario': concat_hue,
                'fwbw': concat_type_rc}
    # Convert the dictionary to a DataFrame
    fig = plt.figure(figsize=(25, 10))
    df = pd.DataFrame(sns_dict)
    row_values = df['fwbw'].unique()
    g = sns.FacetGrid(df, row="fwbw", hue='scenario', height=10, aspect=2, )#gridspec_kws={"hspace":0.0})
    for row_val, ax_g in zip(row_values, g.axes.flat):
        # Filter DataFrame for the current row value
        subset_df = df[df['fwbw'] == row_val]
        # Plot stripplot and pointplot for the current row value
        sns.stripplot(x="x", y="y", hue='scenario', data=subset_df, jitter=True, dodge=True, palette="muted", native_scale=True, alpha=.2, zorder=-1, legend=False, ax=ax_g)
        pointplot = sns.pointplot(x="x", y="y", hue='scenario', data=subset_df, errorbar=('ci', 95), dodge=True, native_scale=True, palette="bright", markersize=6, markeredgewidth=2,  ax=ax_g)
        # Get handles and labels from the last pointplot
        handles, labels =pointplot.get_legend_handles_labels()
    handles, labels = g.axes[0,0].get_legend_handles_labels()
    g.axes[0,0].legend(loc='upper left', bbox_to_anchor=(0.0, 1.05), ncol=6, fontsize="20", frameon=False)
    g.axes[1,0].get_legend().set_visible(False) 
    #Remove titles
    g.set_titles("")
    g.axes[1,0].set_xlabel('True Reserve Capacity', fontsize=22)
    g.axes[0,0].set_ylabel("APE of " + "$R_{M}(-)$", fontsize=22)
    g.axes[1,0].set_ylabel("APE of " + "$R_{M}(+)$", fontsize=22)
    g.axes[0,0].set_aspect('equal')
    g.axes[1,0].set_aspect('equal')
    ace_bins = [1e0, 1e1, 1e2]
    reserve_capacity_bins_gt = np.arange(40, 105, 5)
    for ax_g in g.axes.flat:
        ax_g.set_yscale('log', base=10)  # Apply log scale
        ax_g.set_aspect('auto', adjustable='datalim')  # Manually set the aspect ratio
        ax_g.yaxis.set_major_locator(plt.FixedLocator([1e0, 1e1, 1e2]))  # Set your desired ticks
        ax_g.set_ylim(bottom=1e-0, top=1e2)

    g.axes[0,0].set_xticks(reserve_capacity_bins_gt)
    g.axes[0,0].set_xticklabels([f'{tick/100:.2f}' for tick in reserve_capacity_bins_gt],rotation=30)
    g.axes[1,0].set_xticklabels([f'{tick/100:.2f}' for tick in reserve_capacity_bins_gt],rotation=30, fontsize=20)
    g.axes[0,0].set_yticks(ace_bins)
    g.axes[0,0].set_yticklabels([f'{tick}' for tick in ace_bins], fontsize=20)
    g.axes[1,0].set_yticks(ace_bins)
    g.axes[1,0].set_yticklabels([f'{tick:.2f}' for tick in ace_bins], fontsize=20)
    # Add a big title to the entire plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    g.axes[0,0].tick_params(axis='x', pad=10)
    g.fig.suptitle('Average Percengage Error for ' + (save_dict.split('/')[-1]).upper(), fontsize=20)
    plt.savefig(os.path.join(save_dict, 'mape_vs_rc_' + save_dict.split('/')[-1]+ '_category_' + str(category)+ '.png'))

def err_per_reserve_capacity(error_dict):
    reserve_capacity_bins = np.arange(40, 105, 5)
    for rcb in reserve_capacity_bins:
        err = []
        for scen in error_dict['error']['best_config']['rc+']['scenario'].keys():
            ind = np.where(np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['gt_categ'])== rcb) 
            err.extend(np.array(error_dict['error']['best_config']['rc+']['scenario'][scen]['inst_err'])[ind[0]])
        mean_err =  np.mean(np.array(err))
        std_err = np.std(np.array(err))
        print('RC+ = ', rcb, ' mean error: ', mean_err, 'std : ', std_err)

        err = []
        for scen in error_dict['error']['best_config']['rc-']['scenario'].keys():
            ind = np.where(np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['gt_categ'])== rcb) 
            err.extend(np.array(error_dict['error']['best_config']['rc-']['scenario'][scen]['inst_err'])[ind[0]])
        mean_err =  np.mean(np.array(err))
        std_err = np.std(np.array(err))
        print('RC- = ', rcb, ' mean error: ', mean_err, 'std : ', std_err)
    
    for scen in error_dict['error']['best_config']['rc+']['scenario'].keys():
        print('RC+ Scenario: ', scen, 'mean error: ', error_dict['error']['best_config']['rc+']['scenario'][scen]['mean'],
        'std: ', error_dict['error']['best_config']['rc+']['scenario'][scen]['std'])
        print('RC- Scenario: ', scen, 'mean error: ', error_dict['error']['best_config']['rc-']['scenario'][scen]['mean'],
        'std: ', error_dict['error']['best_config']['rc-']['scenario'][scen]['std'])
    print('RC+ overall mean error: ',  error_dict['error']['best_config']['rc+']['mean'], ' std: ', error_dict['error']['best_config']['rc+']['std'])
    print('RC- overall mean error: ',  error_dict['error']['best_config']['rc-']['mean'], ' std: ', error_dict['error']['best_config']['rc-']['std'])
    print('RC overall mean error: ',  error_dict['error']['best_config']['rc']['mean'], ' std: ', error_dict['error']['best_config']['rc']['std'])

def err_per_reserve_capacity_table_all(methods, method_names, rc_type, file_name, min_rc, threshold=[0.1, 0.25], lprot=['Symmetric', 'Collapse_consistent', 'Monotonic']):
    reserve_capacity_bins = np.arange(min_rc, 105, 5)
    #fill in the first column with the method names
    rcb = 'MAPE' + '\u00B1' + 'Std'
    for th in threshold:
        rcb += ' | Overpred<sub>' + str(th) + '</sub> %'
    rcb = [rcb]+['RC=' + str(rc) for rc in reserve_capacity_bins]
    headerColor = 'grey'
    rowEvenColor = 'white'
    rowOddColor = 'white'
    #results dictionary is the content to be displayed as a table
    results = {}
    results['Method']=rcb
    min_mape = {rcb: [np.inf,-1] for rcb in reserve_capacity_bins}
    min_mape['overall'] = [np.inf,-1] 
    min_overpred = {}
    for th in threshold:
        min_overpred[th] = {rcb: [np.inf,-1]  for rcb in reserve_capacity_bins}
        min_overpred[th]['overall'] = [np.inf,-1] 
    
    for m, mt in enumerate(methods):
        error_dict = np.load(os.path.join(mt, 'error_dict_'+mt+'.npy'), allow_pickle=True).item()
        means = []
        err_overall = []
        #err_overall = np.array(error_dict['error']['best_config'][rc_type]['inst_err'])
        overpred_all = [[] for _ in range(len(threshold))]
        for rcb in reserve_capacity_bins:
            err_rc = []
            overpred = [0]*len(threshold)
            overpred_percentage = [0]*len(threshold)
            total_inst = 0
            for scen in error_dict['error']['best_config'][rc_type]['scenario'].keys():
                lprot_current = scen.split(' ')[0]
                if lprot_current not in lprot:
                    continue
                pred_rc = np.array(error_dict['error']['best_config'][rc_type]['scenario'][scen]['pred'])
                gt_rc = np.array(error_dict['error']['best_config'][rc_type]['scenario'][scen]['gt'])
                inst_err = np.array(error_dict['error']['best_config'][rc_type]['scenario'][scen]['inst_err'])
                ind = np.where(np.array(error_dict['error']['best_config'][rc_type]['scenario'][scen]['gt_categ'])== rcb) 
                pred_rc_selected = pred_rc[ind[0]]
                gt_rc_selected = gt_rc[ind[0]]
                err_rc.extend(inst_err[ind[0]])
                for th_ind, th in enumerate(threshold):
                    overpred[th_ind] += np.sum(pred_rc_selected - gt_rc_selected >= th)
                total_inst += len(pred_rc_selected)

            #this will be used to calculate the avg overpred score across all reserve capacities
            for th_ind, th in enumerate(threshold):
                    if overpred[th_ind] == 0:
                        overpred_all[th_ind].append(overpred[th_ind])
                    else:
                        overpred_percentage[th_ind] = (overpred[th_ind] / total_inst) * 100
                        overpred_all[th_ind].append(overpred_percentage[th_ind])
               
            mean_err =  np.mean(np.array(err_rc))
            std_err = np.std(np.array(err_rc))
            err_overall.append(np.array(err_rc))
            #find the method with the best performance for the current rc level
            if mean_err + std_err < min_mape[rcb][0]:
                min_mape[rcb][0] = mean_err + std_err
                min_mape[rcb][1] = [m]
            elif mean_err + std_err == min_mape[rcb][0]:
                min_mape[rcb][1].append(m)
            #insert the results to the table
            for th_ind, th in enumerate(threshold):
                if overpred_percentage[th_ind] < min_overpred[th][rcb][0]:
                    min_overpred[th][rcb][0] = overpred_percentage[th_ind]
                    min_overpred[th][rcb][1] = [m]
                elif overpred_percentage[th_ind] == min_overpred[th][rcb][0]:
                    min_overpred[th][rcb][1].append(m)
            concatenated_str = f'{mean_err:.2f}\u00B1{std_err:.2f}'
            for th_ind, th in enumerate(threshold):
                concatenated_str += ' | ' f'{overpred_percentage[th_ind]:.2f} % '
            means.append((concatenated_str))
        
        mean_overall = np.mean(np.concatenate(err_overall))
        std_overall = np.std(np.concatenate(err_overall))
            
        #compute avg overpred score across all reserve capacities
        overpred_overall  = {}
        for th_ind, th in enumerate(threshold):
            overpred_overall[th_ind] = (np.mean(np.array(overpred_all[th_ind])))

        means.insert(0, "{:.2f}".format(mean_overall) + '\u00B1' + "{:.2f}".format(std_overall))
        for th_ind, th in enumerate(threshold):
                means[0] += ' | ' + "{:.2f}".format(overpred_overall[th_ind]) + "%"

        #find the method with the best overall performance 
        results[method_names[m]] = means
        if mean_overall  + std_overall < min_mape['overall'][0]:
            min_mape['overall'][0] = mean_overall  + std_overall
            min_mape['overall'][1] = [m]
        elif mean_overall  + std_overall == min_mape['overall'][0]:
            min_mape['overall'][1].append(m)

        for th_ind, th in enumerate(threshold):
            if overpred_overall[th_ind]  < min_overpred[th]['overall'][0]:
                min_overpred[th]['overall'][0] = overpred_overall[th_ind]
                min_overpred[th]['overall'][1] = [m]
            elif overpred_overall[th_ind]  == min_overpred[th]['overall'][0]:
                min_overpred[th]['overall'][1].append(m)
      
    #put the best results in bold
    for rind, rcb in enumerate(reserve_capacity_bins):
        method_mape_best = min_mape[rcb][1]
        method_overpred_best = {}
        for th_ind, th in enumerate(threshold):
            method_overpred_best[th] = min_overpred[th][rcb][1]
        for mmb in method_mape_best:
            result_str = results[method_names[mmb]][rind+1]
            bold_mape = (result_str).split('|')[0]
            results[method_names[mmb]][rind+1] = '<b>' + bold_mape + '</b>'
            for th_ind, th in enumerate(threshold):
                bold_overpred = (result_str).split('|')[th_ind+1]
                results[method_names[mmb]][rind+1] += '|' + bold_overpred
        for th_ind, th in enumerate(threshold):
            for mmb in method_overpred_best[th]:
                result_str = (results[method_names[mmb]][rind+1]).split('|')
                result_str[th_ind+1] = '<b>' + result_str[th_ind+1] + '</b>'
                results[method_names[mmb]][rind+1] = '|'.join(result_str)
                
                    
    method_mape_best = min_mape['overall'][1]
    method_overpred_best = {}
    for th_ind, th in enumerate(threshold):
        method_overpred_best[th] = min_overpred[th]['overall'][1]
    for mmb in method_mape_best:
        result_str = results[method_names[mmb]][0]
        bold_mape = result_str.split('|')[0]
        results[method_names[mmb]][0] = '<b>' + bold_mape + '</b>'
        for th_ind, th in enumerate(threshold):
            bold_overpred = (result_str).split('|')[th_ind+1]
            results[method_names[mmb]][0] += '|' + bold_overpred 
           
    for th_ind, th in enumerate(threshold):
        for mmb in method_overpred_best[th]:
            result_str = results[method_names[mmb]][0].split('|')
            result_str[th_ind+1] = '<b>' + result_str[th_ind+1] + '</b>'
            results[method_names[mmb]][0] = '|'.join(result_str)
    
    width_table=800
    if len(threshold)==1:
        width_table=1200
    elif len(threshold)==2:
        width_table=1400
    fig = go.Figure(data=[go.Table(
    header=dict(
        values=[key.replace('web_', '') for key in results.keys()],
        line_color='darkslategray',
        fill_color=headerColor,
        font=dict(color='white', size=12)
    ),
    cells=dict(
        values=[ results[key] for key in results.keys()],
        line_color='darkslategray',
        # 2-D list of colors for alternating rows
        fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*5],
        font = dict(color = 'black', size = 11)
        ))
    ])
    fig.update_layout(
    autosize=False,
    width=width_table,  # Set the overall width of the table
    height=1000,  # Set the overall height of the table
    margin=dict(l=0, r=0, b=0, t=0),  # Adjust margins as needed
    )
# Specify the column width
    column_widths = [40] + [140]*8
    for i, width in enumerate(column_widths):
        fig.update_layout(
            {'xaxis'+str(i+1): {'dtick': width}}
        )
    #fig.update_layout(width=1500, height=600)
    fig.write_image(file_name + rc_type + ".png", scale=2.0)


if __name__ == '__main__':
   
    #list of methods
    #'knn_drift', 'knn_bottom', 'rf_bottom', 'pnet_bottom', 'resnet_web_bottom', 'resnet_web_bottom_delta', 
    #'resnet_web_bottom', 'pnet_bottom', 'pnet2_bottom', '] #'pnet_bottom_snorm','pnet_bottom_mnorm', 
    #'pnet2_bottom', 'pnet2_bottom_snorm', 'pnet2_bottom_mnorm',]# 'resnet_web_bottom_cnorm', 'resnet_web_bottom_mnorm', 
    #'resnet_web_bottom_delta_cnorm', 'resnet_web_bottom_delta_mnorm'

    #methods = ['resnet_web_bottom_snorm', 'resnet_web_bottom_mnorm', 'pnet_bottom_snorm', 'pnet_bottom_mnorm', 'pnet2_bottom_snorm', 'pnet2_bottom_mnorm']
    #method_names = [ 'Ours Resnet NormS', 'Ours Resnet NormM', 'PointNet NormS', 'PointNet NormM', 'PointNet++ NormS', 'PointNet++ NormM']
    methods = ['pnet2_bottom_snorm_symmetric', 'pnet2_bottom_snorm_collapse_consistent', 'pnet2_bottom_snorm']
    method_names = ['Symmetric', 'Collapse_consistent+Monotonic', 'All']
    #lpror can be one of ['Symmetric', 'Collapse_consistent', 'Monotonic']
    #lprot = ['Symmetric', 'Collapse_consistent', 'Monotonic']
    lprot = ['Monotonic']
    min_rc = 40 #40
    err_per_reserve_capacity_table_all(methods, method_names, 'rc+', 'err_w_monotonic', min_rc, threshold=[0.1, 0.25], lprot=lprot)
    err_per_reserve_capacity_table_all(methods, method_names, 'rc-', 'err_w_monotonic', min_rc, threshold=[0.1, 0.25], lprot=lprot)

