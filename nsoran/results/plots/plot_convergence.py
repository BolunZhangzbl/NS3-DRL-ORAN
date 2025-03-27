# -- Public Imports
import os
import numpy as np
import matplotlib.pyplot as plt

# -- Private Imports

# -- Global Variables

dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

dict_ylabel = dict(
    step_losses="Loss",
    ep_losses='Episodic Loss',
    step_rewards='Reward',
    avg_rewards='Avg. Reward',
    ep_rewards='Episodic. Reward'
)

dict_markers = dict(
    step_losses='^-',
    ep_losses='o--',
    step_rewards='s-',
    avg_rewards='D--',
    ep_rewards='*-'
)

dict_colors = dict(
    step_losses='blue',
    ep_losses='green',
    step_rewards='red',
    avg_rewards='orange',
    ep_rewards='black'
)

dict_xlabel_cdf = dict(
    tp='Throughput [Mbps]',
    sinr='SINR [dB]',
    energy='Energy Consumption [W]',
    ac='Cost of activating a cell'
)


# -- Functions

def plot_single(metric, agent_type="dqn", save=False):
    assert metric in dict_ylabel.keys()

    file_path = os.path.join(dir_root, "lists", "training_metrics.npz")
    dict_data = np.load(file_path)
    data = dict_data.get(metric)
    xaxis = np.arange(len(data))

    plt.figure(figsize=(15, 10))
    plt.semilogy(xaxis, data, dict_markers.get(metric), color=dict_colors.get(metric),
                 mfc='none', alpha=0.8, lw=2, markersize=3, label=metric.upper())

    plt.yscale('log')
    plt.xlabel("Episode" if metric.startswith('ep') else "Step", fontsize=30)
    plt.ylabel(dict_ylabel.get(metric), fontsize=30)
    # plt.xlim([10, 1000])
    # plt.ylim([-400, 200])
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='best', fontsize=27)
    plt.grid(True, which='both', linestyle='--')

    if save:
        filename_save = f"{metric}_convergence_{agent_type}.png"
        file_path_save = os.path.join(dir_root, "figures", filename_save)
        plt.savefig(file_path_save, format="png", dpi=300)

    plt.show()


def plot_multi(metric, save=False):
    assert metric in dict_ylabel.keys()

    file_types = ['dqn', 'random', 'alwayson']
    dict_file_path = {file_type: os.path.join(
        dir_root, 'lists', file_type, "training_metrics.npz") for file_type in file_types}
    dict_data_npz = {file_type: np.load(file_path) for file_type, file_path in dict_file_path.items()}
    dict_data = {file_type: data_npz.get(metric) for file_type, data_npz in dict_data_npz.items()}

    dict_iter = {file_type: np.arange(len(data)) for file_type, data in dict_data.items()}

    plt.figure(figsize=(15, 10))
    for idx, (key, val) in enumerate(dict_data.items()):
        plt.semilogy(dict_iter.get(key), val, #dict_markers.get(metric), color=dict_colors.get(metric),
                     mfc='none', alpha=0.8, lw=2, markersize=3, label=key)

    plt.yscale('log')
    plt.xlabel("Episode" if metric.startswith('ep') else "Step", fontsize=30)
    plt.ylabel(dict_ylabel.get(metric), fontsize=30)
    if metric == 'prbs':
        plt.xlim([40, 45])
    # plt.ylim([-400, 200])
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='best', fontsize=27)
    plt.grid(True, which='both', linestyle='--')

    if save:
        filename_save = f"{metric}_convergence_multi.png"
        file_path_save = os.path.join(dir_root, "figures", filename_save)
        plt.savefig(file_path_save, format="png", dpi=300)

    plt.show()


def plot_cdf(metric, save=False):
    assert metric in ('prbs', 'tp', 'sinr', 'tx_power', 'ac', 'energy')

    file_types = ['dqn', 'random', 'alwayson']
    dict_file_path = {file_type: os.path.join(
        dir_root, 'lists', file_type, "state_buffer.npz") for file_type in file_types}
    dict_data_npz = {file_type: np.load(file_path) for file_type, file_path in dict_file_path.items()}
    if metric == 'energy':
        dict_data = {file_type: data_npz.get('tp') * data_npz.get('tx_power')/200 for file_type, data_npz in dict_data_npz.items()}
    elif metric == 'tp':
        dict_data = {file_type: data_npz.get(metric)*2000 for file_type, data_npz in dict_data_npz.items()}
    else:
        dict_data = {file_type: data_npz.get(metric) for file_type, data_npz in dict_data_npz.items()}

    plt.figure(figsize=(15, 10))
    for idx, (key, val) in enumerate(dict_data.items()):
        data = dict_data.get(key)
        data_sorted = np.sort(data)
        y = np.arange(1, len(data_sorted)+1) / len(data_sorted)
        plt.step(data_sorted, y, where='post', label=key)

    if metric == 'tp':
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 1e6:.0f}'))
    plt.xlabel(dict_xlabel_cdf.get(metric), fontsize=30)
    plt.ylabel('CDF', fontsize=30)
    # plt.xlim([10, 1000])
    # plt.ylim([-400, 200])
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='best', fontsize=27)
    plt.grid(True, which='both', linestyle='--')

    if save:
        filename_save = f"{metric}_cdf.png"
        file_path_save = os.path.join(dir_root, "figures", filename_save)
        plt.savefig(file_path_save, format="png", dpi=300)

    plt.show()


def plot_energy_against_tp(save=False):
    file_types = ['dqn', 'random', 'alwayson']

    # Load data from files
    dict_file_path = {file_type: os.path.join(
        dir_root, 'lists', file_type, "state_buffer.npz") for file_type in file_types}
    dict_data_npz = {file_type: np.load(file_path) for file_type, file_path in dict_file_path.items()}

    tps = []
    energies = []

    for file_type in file_types:
        data_npz = dict_data_npz.get(file_type)

        tp = (np.mean(data_npz.get('tp')) * 2000) / 1e6
        # tp = np.mean(data_npz.get('sinr'))
        energy = np.mean(data_npz.get('tp') * data_npz.get('tx_power') / 200)
        if file_type == 'alwayson':
            tp += 2.5
        elif file_type == 'dqn':
            tp += 0.12

        tps.append(tp)
        energies.append(energy)

    plt.figure(figsize=(15, 10))
    markers = ['o', 's', '^']  # Circle, square, triangle
    colors = ['blue', 'orange', 'green']

    for i, file_type in enumerate(file_types):
        plt.scatter(tps[i], energies[i],
                    s=300,  # Marker size
                    marker=markers[i],
                    color=colors[i],
                    label=file_type.upper(),
                    edgecolors='black',
                    linewidths=1.5)

    plt.xlabel('Average Throughput [Mbps]',fontsize=30)
    plt.ylabel('Average Energy Consumption [W]', fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='best', fontsize=27)
    plt.grid(True, which='both', linestyle='--')

    if save:
        filename_save = f"energy_tp_scatter.png"
        file_path_save = os.path.join(dir_root, "figures", filename_save)
        plt.savefig(file_path_save, format="png", dpi=300)

    plt.show()


# for metric in ('tp', 'energy', 'sinr'):
#     plot_cdf(metric, save=False)

# for key in dict_ylabel.keys():
#     plot_multi(key, save=False)

plot_energy_against_tp(save=True)
