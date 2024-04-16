#  :target: https://mybinder.org/v2/gh/MauricioSalazare/tensorpowerflow/master?filepath=examples

from tensorpowerflow import GridTensor
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TKAgg')

# Load 2 years of data at 30 min resolution (35040 entries)
print("Loading data...")
active_power_time_series = pd.read_csv("data/time_series/active_power_example.csv",
                                       index_col=0,
                                       parse_dates=True)
reactive_power_time_series = pd.read_csv("data/time_series/reactive_power_example.csv",
                                         index_col=0,
                                         parse_dates=True)

#%% Inspect data
print("Inspecting data...")
fig, ax = plt.subplots(2, 1, figsize=(10, 5))
plt.subplots_adjust(hspace=0.5, bottom=0.15, top=0.95)
active_power_time_series.head(48 * 7).plot(ax=ax[0], legend=False, color="b")
reactive_power_time_series.head(48 * 7).plot(ax=ax[1], legend=False, color="r")
ax[0].set_ylabel("Active power [kW]")
ax[1].set_ylabel("Reactive power [kVAr]")
ax[0].set_xlabel("")
ax[1].set_xlabel("")

#%% Create the grid
network = GridTensor(node_file_path="data/grid_data/Nodes_34.csv",
                     lines_file_path="data/grid_data/Lines_34.csv",
                     gpu_mode=False)

#%% Run the power flow
print("Solving power flow...")
solutions = network.run_pf(active_power=active_power_time_series.values,
                           reactive_power=reactive_power_time_series.values)

#%% Plot results of voltage in one node
ylabels = ["Volt. mag\n[p.u.]", "Active power\n[kW]", "Reactive power\n[kVAr]"]
fig, ax = plt.subplots(3, 1, figsize=(10, 5))
plt.subplots_adjust(hspace=0.5, bottom=0.1, top=0.95)
ax[0].plot(np.abs(solutions["v"][:(48 * 7), 0]), label="Node 2", color="green")
ax[0].plot(np.abs(solutions["v"][:(48 * 7), 32]), label="Node 34", color="k")
ax[1].plot(active_power_time_series.values[:(48 * 7), 0], label="Node 2", color="green")
ax[1].plot(active_power_time_series.values[:(48 * 7), 32], label="Node 34", color="k")
ax[2].plot(reactive_power_time_series.values[:(48 * 7), 0], label="Node 2", color="green")
ax[2].plot(reactive_power_time_series.values[:(48 * 7), 32], label="Node 34", color="k")
ax[0].set_ylim((0.94, 1.01))
for ax_, ylabel_ in zip(ax, ylabels):
    ax_.legend(loc="upper right", fontsize="xx-small")
    ax_.set_ylabel(ylabel_, fontsize="small")


