import ast
import pandas as pd
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
from distutils.version import LooseVersion

"""
Computes the plot for the journal.
The sweep_experiments.py should be run first to have the data for the plots.
"""


#%%
def set_figure_art(fontsize=7, usetex=False):
    # fontsize = 7
    linewidth_axes = 0.4
    linewidth_grid = 0.2

    linewidth_ticks_major = 0.5
    linewidth_ticks_minor = 0.1

    linewidth_lines = 0.8
    # usetex=True
    matplotlib.rc('legend', fontsize=fontsize, handlelength=3)
    matplotlib.rc('axes', titlesize=fontsize)
    matplotlib.rc('axes', labelsize=fontsize)
    matplotlib.rc('axes', linewidth=linewidth_axes)
    matplotlib.rc('patch', linewidth=linewidth_axes)
    matplotlib.rc('hatch', linewidth=linewidth_axes)
    matplotlib.rc('xtick', labelsize=fontsize)
    matplotlib.rc('xtick.major', width=linewidth_ticks_major)
    matplotlib.rc('ytick.major', width=linewidth_ticks_major)
    matplotlib.rc('xtick.minor', width=linewidth_ticks_minor)
    matplotlib.rc('ytick.minor', width=linewidth_ticks_minor)
    matplotlib.rc('ytick', labelsize=fontsize)
    matplotlib.rc('lines', linewidth=linewidth_lines)
    matplotlib.rc('text', usetex=usetex)
    matplotlib.rc('font', size=fontsize, family='serif',
                  style='normal', variant='normal',
                  stretch='normal', weight='normal')
    matplotlib.rc('patch', force_edgecolor=True)
    if LooseVersion(matplotlib.__version__) < LooseVersion("3.1"):
        matplotlib.rc('_internal', classic_mode=True)
    else:
        # New in mpl 3.1
        matplotlib.rc('scatter', edgecolors='b')
    matplotlib.rc('grid', linestyle=':', linewidth=linewidth_grid)
    matplotlib.rc('errorbar', capsize=3)
    matplotlib.rc('image', cmap='viridis')
    matplotlib.rc('axes', xmargin=0)
    matplotlib.rc('axes', ymargin=0)
    matplotlib.rc('xtick', direction='in')
    matplotlib.rc('ytick', direction='in')
    matplotlib.rc('xtick', top=True)
    matplotlib.rc('ytick', right=True)
    # rcdefaults() # Reset the default settings of Matplotlib
    # plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])

#%%

def line_plot(tables_tests_concat: pd.DataFrame,
              time_steps: int,
              time_column: str,
              ax):
    colors = cycle(["red", "blue", "green", "olive", "purple", "fuchsia", "peru"])
    line_styles = cycle(["-", "--", "-.", ":", "-", "--"])
    marker_styles = cycle([".", ",", "o", "v", "<", ">", "1"])

    filtered_by_time_step = tables_tests_concat.loc[tables_tests_concat["time_steps"] == time_steps].copy()
    method_names = filtered_by_time_step["index"].unique()

    for method_name, _color, _marker, _linestyle in zip(method_names, colors, marker_styles, line_styles):
        filtered_by_method = filtered_by_time_step.loc[filtered_by_time_step["index"] == method_name]
        ax.plot(filtered_by_method["grid_size"], filtered_by_method[time_column],
                  marker=_marker,
                  color=_color,
                  linestyle=_linestyle,
                  label=method_name)

def read_data(file_name):
    test_solutions = pd.read_csv(file_name)

    if "Unnamed: 0" in test_solutions.columns:
        test_solutions.rename(columns={"Unnamed: 0": "index"}, inplace=True)

    columns_to_process = ['iterations_log', 'time_pre_pf_log', 'time_pf_log', 'convergence_log']
    for _process_column in columns_to_process:
        test_solutions[_process_column] = test_solutions[_process_column].apply(lambda x: ast.literal_eval(x))

    return test_solutions


def asympotic_compexity(n, c, k):
    return c * np.power(n, k)

def fit_parameters(data_frame: pd.DataFrame):
    # Code taken from:
    # https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly

    method_names = data_frame["index"].unique()
    grid_sizes = data_frame["grid_size"].unique()
    fitting_parameters = {}

    for method_name in method_names:
        fitting_parameters[method_name] = {}

        for grid_size in grid_sizes:
            filtered_by_grid_size = data_frame.loc[data_frame["grid_size"] == grid_size].copy()
            filtered_frame_by_method = filtered_by_grid_size.loc[data_frame["index"] == method_name]

            n = filtered_frame_by_method["time_steps"].values.astype(float)
            tc = filtered_frame_by_method["time_pf"].values.astype(float)
            popt, _ = curve_fit(asympotic_compexity, n, tc)

            fitting_parameters[method_name].update({grid_size: {"c": popt[0],
                                                                "k": popt[1]}})

            # print(f"Complexity '{method_name}': c={popt[0]:.4f}, k={popt[1]:.4f}")

    fitting_parameters_frame = pd.concat({k: pd.DataFrame(v).T for k, v in fitting_parameters.items()}, axis=0)
    fitting_parameters_frame.reset_index(inplace=True)
    fitting_parameters_frame = fitting_parameters_frame.rename(columns={"level_0": "algorithm", "level_1": "grid_size"})

    # print(fitting_parameters_frame.groupby("algorithm").mean().sort_values(by=["k"]).drop(columns=["grid_size"]))

    return fitting_parameters_frame


def figure_1(data_frame: pd.DataFrame, time_steps: int):
    # Plot total time of the experiments

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    ax1 = ax[0, :].flatten()
    ax2 = ax[1, :].flatten()
    times_columns = ['time_pre_pf', 'time_pf', 'time_algorithm']

    for time_column, ax1_, ax2_ in zip(times_columns, ax1, ax2):
        # filtered_by_time_step = data_frame.loc[data_frame["time_steps"] == time_steps].copy()
        # method_names = filtered_by_time_step["index"].unique()

        line_plot(tables_tests_concat=data_frame,
                  time_steps=time_steps,
                  time_column=time_column,
                  ax=ax1_)
        ax1_.set_xscale('log')
        ax1_.set_yscale('log')
        ax1_.set_title(time_column)
        ax1_.grid(which="minor", linestyle="--", linewidth=0.4)
        ax1_.grid(which="major", linestyle="-", linewidth=0.8)

        ax1_.set_xlabel("Grid size nodes [" + r"$n$" + "]")
        ax1_.set_ylabel("Time power flow [sec]")
        ax1_.legend(fontsize="x-small")

        line_plot(tables_tests_concat=data_frame,
                  time_steps=time_steps,
                  time_column=time_column,
                  ax=ax2_)

        ax2_.legend(fontsize="x-small")
        ax2_.set_title(time_column)
        ax2_.grid(which="minor", linestyle="--", linewidth=0.4)
        ax2_.grid(which="major", linestyle="-", linewidth=0.8)
        ax2_.set_xlabel("Grid size nodes [" + r"$n$" + "]")
        ax2_.set_ylabel("Time power flow [sec]")
    fig.suptitle(f"Total Power Flows: {time_steps}")

    return ax
    # plt.savefig("figures/grid_size_sweep.pdf")

def figure_2(data_frame: pd.DataFrame, time_steps: int):
    # Plot total time of the experiments including bootstrap
    unique_index = data_frame["index"].unique()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.subplots_adjust(left=0.15)
    w = 0.1
    width = lambda p, w: 10 ** (np.log10(p) + w / 2.) - 10 ** (np.log10(p) - w / 2.)

    colors = cycle(["red", "blue", "green", "olive", "purple", "fuchsia", "peru"])
    line_styles = cycle(["-", "--", "-.", ":", "-", "--"])
    marker_styles = cycle([".", ",", "o", "v", "<", ">", "1"])

    filtered_by_time_step = data_frame.loc[data_frame["time_steps"] == time_steps].copy()

    for method_name, _color, _linestyle, _markerstyle in zip(unique_index, colors, line_styles, marker_styles):
        filtered_frame_by_method = filtered_by_time_step.loc[data_frame["index"] == method_name]
        grid_size_list = filtered_frame_by_method["grid_size"].unique()

        data_list = []
        for _grid_size in grid_size_list:
            idx = filtered_frame_by_method["grid_size"] == _grid_size
            data_list.append(filtered_frame_by_method.loc[idx]["time_pf"].values)

        line_median = [np.median(x) for x in data_list]

        median_properties = dict(color=_color)
        box_artist = ax.boxplot(data_list,
                                positions=grid_size_list,
                                widths=width(grid_size_list, w),
                                showfliers=False,
                                medianprops=median_properties, )
        # flierprops=flier_properties)
        ax.plot(grid_size_list, line_median,
                color=_color,
                marker=_markerstyle,
                linestyle=_linestyle,
                label=method_name)

    # ax.set_xlim(4, 4000)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(which="minor", linestyle="--", linewidth=0.4)
    ax.grid(which="major", linestyle="-", linewidth=0.8)
    ax.set_xticks(grid_size_list)
    ax.set_ylabel("Time power flow [sec]")
    ax.set_xlabel("Grid size")
    ax.set_title(f"Statitics comp. time {time_steps} power flows")
    ax.legend(fontsize="x-small")


def figure_3(data_frame: pd.DataFrame, grid_size: int):
    """Time steps sweep"""

    unique_index = data_frame["index"].unique()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.subplots_adjust(left=0.15)
    w = 0.1
    width = lambda p, w: 10 ** (np.log10(p) + w / 2.) - 10 ** (np.log10(p) - w / 2.)

    colors = cycle(["red", "blue", "green", "olive", "purple", "fuchsia", "peru"])
    line_styles = cycle(["-", "--", "-.", ":", "-", "--"])
    marker_styles = cycle([".", ",", "o", "v", "<", ">", "1"])

    filtered_by_grid_size = data_frame.loc[data_frame["grid_size"] == grid_size].copy()

    for method_name, _color, _linestyle, _markerstyle in zip(unique_index, colors, line_styles, marker_styles):
        filtered_frame_by_method = filtered_by_grid_size.loc[data_frame["index"] == method_name]
        time_steps_list = filtered_frame_by_method["time_steps"].unique()

        data_list = []
        for _time_step in time_steps_list:
            idx = filtered_frame_by_method["time_steps"] == _time_step
            data_list.append(filtered_frame_by_method.loc[idx]["time_pf"].values)

        line_median = [np.median(x) for x in data_list]

        median_properties = dict(color=_color)
        box_artist = ax.boxplot(data_list,
                                positions=time_steps_list,
                                widths=width(time_steps_list, w),
                                showfliers=False,
                                medianprops=median_properties, )
        # flierprops=flier_properties)
        ax.plot(time_steps_list, line_median,
                color=_color,
                marker=_markerstyle,
                linestyle=_linestyle,
                label=method_name)

    # ax.set_xlim(0.1, 20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(which="minor", linewidth=0.4, dashes=(1,10,1,10))
    ax.grid(which="major", linestyle="-", linewidth=0.8)
    ax.set_xticks(time_steps_list)
    ax.set_ylabel("Time power flow [sec]")
    ax.set_xlabel("Time steps")
    ax.set_title(f"Statitics Grid size {grid_size}")
    ax.legend(fontsize="x-small")

    return ax


def figure_4(data_frame):
    fitting_parameters_frame = fit_parameters(data_frame)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    algorithm_names = fitting_parameters_frame["algorithm"].unique()
    colors = cycle(["red", "blue", "green", "olive", "purple", "fuchsia", "peru"])
    ax2 = ax.twinx()
    for algorithm_name, color in zip(algorithm_names, colors):
        filtered_frame_by_algorithm = fitting_parameters_frame.loc[
            fitting_parameters_frame["algorithm"] == algorithm_name]
        ax.plot(filtered_frame_by_algorithm["grid_size"].values,
                filtered_frame_by_algorithm["c"].values,
                linestyle="-",
                color=color,
                label="'c' " + algorithm_name)
        ax2.plot(filtered_frame_by_algorithm["grid_size"].values,
                 filtered_frame_by_algorithm["k"].values,
                 linestyle="--",
                 color=color,
                 label="'k' " + algorithm_name)
    ax.set_ylabel("c")
    ax2.set_ylabel("k")
    ax.set_title("tc = c * (n ^ k)")
    ax.legend(fontsize="x-small", loc="upper center")
    ax2.legend(fontsize="x-small", loc="center right")
    # ax.grid(which="major", linestyle="--", linewidth=0.8)
    # ax2.grid(which="major", linestyle="-", linewidth=0.9)
    ax.set_xlabel("Grid size")

def unmelted_dataframe(df, algorithm, bootstrap):
    filter_1 = df.loc[(df["index"] == algorithm) & (df["bootstrap"] == bootstrap)]
    filter_2 = filter_1[["time_pf", "time_steps", "grid_size"]].copy()
    df_unmelted = filter_2.pivot(index="time_steps", columns="grid_size", values="time_pf")

    return df_unmelted

def plotly_figure(data_frame, algorithm: str = "tensor", bootstrap: int = 1):
    import plotly.graph_objects as go
    import pandas as pd
    import plotly.io as pio
    pio.renderers.default = "browser"


    cmap = plt.get_cmap("seismic")

    # (r,g,b,alpha)<- cmap(1)
    colorscale1 = [[0, 'rgb' + str(cmap(1)[0:3])],
                  [1, 'rgb' + str(cmap(1)[0:3])]]

    colorscale2 = [[0, 'rgb' + str(cmap(255)[0:3])],
                  [1, 'rgb' + str(cmap(255)[0:3])]]


    colorscale3 = [[0, 'rgb' + str(cmap(122)[0:3])],
                  [1, 'rgb' + str(cmap(122)[0:3])]]

    colorscale4 = [[0, 'rgb' + "(0.709, 0.701, 0.360)"],
                  [1, 'rgb' + "(0.709, 0.701, 0.360)"]]

    # Read data from a csv
    df_unmelted1 = unmelted_dataframe(df=data_frame, algorithm=algorithm, bootstrap=bootstrap)
    df_unmelted2 = unmelted_dataframe(df=data_frame, algorithm="sequential", bootstrap=bootstrap)
    df_unmelted3 = unmelted_dataframe(df=data_frame, algorithm="hp-pardiso", bootstrap=bootstrap)
    df_unmelted4 = unmelted_dataframe(df=data_frame, algorithm="hp-tensor", bootstrap=bootstrap)
    colors_saddle = np.ones(shape=df_unmelted1.values.shape)

    fig = go.Figure(data=[go.Surface(z=df_unmelted1.values,
                                     x=df_unmelted1.index.values,
                                     y=df_unmelted1.columns.values,
                                     # surfacecolor=colors_saddle,
                                     name="Tensor",
                                     opacity=1.0,
                                     cmin=0,
                                     cmax=1,
                                     colorscale=colorscale1,
                                     showscale=False,
                                     showlegend=True),
                          go.Surface(z=df_unmelted2.values,
                                     x=df_unmelted2.index.values,
                                     y=df_unmelted2.columns.values,
                                     # surfacecolor=colors_saddle,
                                     name="Sequential",
                                     opacity=1.0,
                                     cmin=0,
                                     cmax=1,
                                     colorscale=colorscale2,
                                     showscale=False,
                                     showlegend=True),
                          go.Surface(z=df_unmelted3.values,
                                     x=df_unmelted3.index.values,
                                     y=df_unmelted3.columns.values,
                                     # surfacecolor=colors_saddle,
                                     name="HP-pardiso-Secuencial",
                                     opacity=1.0,
                                     cmin=0,
                                     cmax=1,
                                     colorscale=colorscale3,
                                     showscale=False,
                                     showlegend=True),
                          go.Surface(z=df_unmelted4.values,
                                     x=df_unmelted4.index.values,
                                     y=df_unmelted4.columns.values,
                                     # surfacecolor=colors_saddle,
                                     name="HP-Tensor",
                                     opacity=1.0,
                                     cmin=0,
                                     cmax=1,
                                     colorscale=colorscale4,
                                     showscale=False,
                                     showlegend=True)
                          ])

    fig.update_layout(title='Algorithm performance', autosize=False,
                      width=800, height=800,
                      margin=dict(l=100, r=50, b=65, t=90),
                      scene=dict(xaxis=dict(type="log", dtick=1),
                                 yaxis=dict(type="log", dtick=1),
                                 zaxis=dict(type="log", dtick=1),
                                 aspectmode="manual",
                                 aspectratio=dict(x=1, y=1, z=0.8),
                                 xaxis_title = 'Time steps',
                                 yaxis_title = 'Grid size',
                                 zaxis_title = 'Comp. time [sec]',
                                )
                      )
    fig.show()



#%%
if __name__ == "__main__":

    #%%
    # file_name_ts_sweep = r"data\test_ts_fixed_grid_500_DO_NOT_DELETE.csv"
    file_name_ts_sweep = r"data\merged_data_plot.csv"
    test_solutions_ts_sweep = read_data(file_name=file_name_ts_sweep)

    # file_name_grid_sweep = r"data\test_grid_sweep_two_time_steps_1_1000_DO_NOT_DELETE.csv"
    # file_name_grid_sweep = r"data\test_grid_sweep_two_time_steps_1_1000_DO_NOT_DELETE.csv"
    file_name_grid_sweep =  r"data\merged_data_plot.csv"
    # file_name_grid_sweep = r"data\test_grid_sweep_1_1000.csv"
    # file_name_grid_sweep = r"sam_test.csv"
    # file_name_grid_sweep = r"delete_me.csv"
    # file_name_grid_sweep = r"delete_me_numba.csv"
    test_solutions_grid_sweep = read_data(file_name=file_name_grid_sweep)

    # file_name_ts_sweep = r"data\test_parameter_calculation_DO_NOT_DELETE.csv"
    file_name_ts_sweep = r"data\merged_data_plot.csv"
    test_solutions_parameter = read_data(file_name=file_name_ts_sweep)


    ax_fig_1 = figure_1(test_solutions_grid_sweep, time_steps=1)  # Grid sweep at fixed time step
    ax_fig_2 = figure_1(test_solutions_grid_sweep, time_steps=1000)  # Grid sweep at fixed time step

    print("1 time step:")
    print(f"xticks major: {ax_fig_1[0, 1].get_xticks()}")
    print(f"xticks minor: {ax_fig_1[0, 1].xaxis.get_ticklocs(minor=True)}")

    ts_1_minor_xticks = ax_fig_1[0, 1].xaxis.get_ticklocs(minor=True)
    ts_1_minor_yticks = ax_fig_1[0, 1].yaxis.get_ticklocs(minor=True)

    ts_1_major_xticks = ax_fig_1[0, 1].xaxis.get_ticklocs(minor=False)
    ts_1_major_yticks = ax_fig_1[0, 1].yaxis.get_ticklocs(minor=False)

    print(f"xticks_labels: {ax_fig_1[0, 1].get_xticklabels()}")
    print(f"xlim: {ax_fig_1[0, 1].get_xlim()}\n")

    print(f"yticks: {ax_fig_1[0, 1].get_yticks()}")
    print(f"yticklables: {ax_fig_1[0, 1].get_yticklabels()}")
    print(f"ylim: {ax_fig_1[0, 1].get_ylim()}\n\n")

    print("1000 time steps:")
    print(f"xticks: {ax_fig_2[0, 1].get_xticks()}")
    print(f"xticks minor: {ax_fig_2[0, 1].xaxis.get_ticklocs(minor=True)}")

    ts_1000_minor_xticks = ax_fig_2[0, 1].xaxis.get_ticklocs(minor=True)
    ts_1000_minor_yticks = ax_fig_2[0, 1].yaxis.get_ticklocs(minor=True)

    ts_1000_major_xticks = ax_fig_2[0, 1].xaxis.get_ticklocs(minor=False)
    ts_1000_major_yticks = ax_fig_2[0, 1].yaxis.get_ticklocs(minor=False)

    print(f"xticks_labels: {ax_fig_2[0, 1].get_xticklabels()}")
    print(f"xlim: {ax_fig_2[0, 1].get_xlim()}\n")

    print(f"yticks: {ax_fig_2[0, 1].get_yticks()}")
    print(f"yticklables: {ax_fig_2[0, 1].get_yticklabels()}")
    print(f"ylim: {ax_fig_2[0, 1].get_ylim()}\n\n")

    # figure_2(test_solutions_grid_sweep, time_steps=10)  # Grid sweep with bootstrap at fixed time step
    ax_fig_3 = figure_3(data_frame=test_solutions_ts_sweep, grid_size=500)  # Time steps sweep at fixed grid_size
    print("Time sweep:")
    print(f"xticks: {ax_fig_3.get_xticks()}")
    print(f"xticks_labels: {ax_fig_3.get_xticklabels()}")
    print(f"xlim: {ax_fig_3.get_xlim()}\n")

    print(f"yticks: {ax_fig_3.get_yticks()}")
    print(f"yticklables: {ax_fig_3.get_yticklabels()}")
    print(f"ylim: {ax_fig_3.get_ylim()}")

    time_sweep_major_yticks = ax_fig_3.yaxis.get_ticklocs(minor=False)
    time_sweep_major_xticks = ax_fig_3.xaxis.get_ticklocs(minor=False)

    time_sweep_minor_xticks = ax_fig_3.xaxis.get_ticklocs(minor=True)
    time_sweep_minor_yticks = ax_fig_3.yaxis.get_ticklocs(minor=True)


    time_sweep_xlim = ax_fig_3.get_xlim()
    time_sweep_ylim = ax_fig_3.get_ylim()

    # import matplotlib as mpl
    #
    # mpl.rc_context()

    figure_4(test_solutions_parameter)  # Plot of parameter fitting

    #%% Plots for the journal:

    # Figure 1:
    #                         |
    #  Grid size sweep 1 TS   | Grid size sweep 1000 TS
    #  -----------------------------------------------
    #  TS Sweep grid size=500 |  parameters
    #                         |

    # Figure art:
    set_figure_art(fontsize=6)
    linewidth_minor_grid = 0.1
    linewidth_major_grid = 0.3
    dashes_minor_grid = (1, 10, 1, 10)
    marker_size = 1

    # Prepare data for grid size sweep:
    ts_filter = [1, 1000]
    data_filtered = []
    for ts in ts_filter:
        data_filtered.append(test_solutions_grid_sweep.loc[test_solutions_grid_sweep["time_steps"] == ts].copy())
    method_names = test_solutions_grid_sweep["index"].unique()

    # Prepare data for time step sweep:
    filtered_by_grid_size = test_solutions_ts_sweep.loc[test_solutions_ts_sweep["grid_size"] == 500].copy()

    # Prepare data for parameter fit:
    fitting_parameters_frame = fit_parameters(data_frame=test_solutions_parameter)


    title_label = ["(a)", "(b)"]
    cm = 25/64

    fig, ax = plt.subplots(2, 2, figsize=(9*cm, 9*cm))
    plt.subplots_adjust(left=0.15, right=0.9, wspace=0.45, hspace=0.6, top=0.95, bottom=0.22)
    ax0 = ax[0,:].flatten()
    ax1 = ax[1, :].flatten()

    # ==================================================================================================================
    # Upper row subplots (First row, 2 columns)
    # ==================================================================================================================
    for ii, (ax_col, filtered_by_time_step, title) in enumerate(zip(ax0, data_filtered, title_label)):
        # Reset colors
        colors = cycle(["red", "blue", "green", "olive", "purple", "fuchsia", "peru"])
        line_styles = cycle(["-", "--", "-.", ":", "-", "--"])
        marker_styles = cycle([".", ",", "o", "v", "<", ">", "1"])

        for method_name, _color, _marker, _linestyle in zip(method_names, colors, marker_styles, line_styles):
            filtered_by_method = filtered_by_time_step.loc[filtered_by_time_step["index"] == method_name]
            ax_col.plot(filtered_by_method["grid_size"], filtered_by_method["time_pf"],
                           marker=_marker,
                           color=_color,
                           linestyle=_linestyle,
                           linewidth=0.5,
                           markersize=marker_size,
                           label=method_name)

        ax_col.tick_params(axis='both', which='major', labelsize=6)
        ax_col.set_xscale('log')
        ax_col.set_yscale('log')
        ax_col.set_title(title, fontsize=6)
        ax_col.grid(which="minor", linewidth=linewidth_minor_grid, dashes=dashes_minor_grid)
        ax_col.grid(which="major", linestyle="-", linewidth=linewidth_major_grid)
        ax_col.set_xlim((99, 5001))

        ax_col.set_xlabel("Grid size nodes [" + r"$n$" + "]", fontsize=6)
        if ii == 0:
            ax_col.set_ylabel("Time power flow \n[sec]", fontsize=6)

    # Plot of 1 time step:
    ax[0, 0].set_xticks(np.logspace(np.log10(0.1), np.log10(1e5), 10, endpoint=True))
    ax[0, 0].xaxis.set_ticks(ts_1_minor_xticks, minor=True)
    ax[0, 0].xaxis.set_ticks(ts_1_major_xticks, minor=False)
    ax[0, 0].set_xlim((9, 6000))

    ax[0, 0].set_yticks(np.logspace(np.log10(1e-6), np.log10(1e2), 9, endpoint=True))
    ax[0, 0].yaxis.set_ticks(ts_1_minor_yticks, minor=True)
    ax[0, 0].yaxis.set_ticks(ts_1_major_yticks, minor=False)
    ax[0, 0].set_ylim((9e-5, 2))

    # Plot of 1000 time steps
    ax[0, 1].set_xticks(np.logspace(np.log10(0.1), np.log10(1e5), 10, endpoint=True))
    ax[0, 1].xaxis.set_ticks(ts_1000_minor_xticks, minor=True)
    ax[0, 1].xaxis.set_ticks(ts_1000_major_xticks, minor=False)
    ax[0, 1].set_xlim((9, 6000))

    ax[0, 1].set_yticks(np.logspace(np.log10(1e-4), np.log10(1e5), 10, endpoint=True))
    ax[0, 1].yaxis.set_ticks(ts_1000_minor_yticks, minor=True)
    ax[0, 1].yaxis.set_ticks(ts_1000_major_yticks, minor=False)
    ax[0, 1].set_ylim((3e-3, 1300))


    # ==================================================================================================================
    # Third subplot (Second row, first column)
    # ==================================================================================================================
    # Reset colors
    colors = cycle(["red", "blue", "green", "olive", "purple", "fuchsia", "peru"])
    line_styles = cycle(["-", "--", "-.", ":", "-", "--"])
    marker_styles = cycle([".", ",", "o", "v", "<", ">", "1"])

    for method_name, _color, _linestyle, _markerstyle in zip(method_names, colors, line_styles, marker_styles):
        filtered_frame_by_method = filtered_by_grid_size.loc[filtered_by_grid_size["index"] == method_name]
        time_steps_list = filtered_frame_by_method["time_steps"].unique()

        data_list = []
        for _time_step in time_steps_list:
            idx = filtered_frame_by_method["time_steps"] == _time_step
            data_list.append(filtered_frame_by_method.loc[idx]["time_pf"].values)


        ax1[0].plot(filtered_frame_by_method["time_steps"].to_list(),
                filtered_frame_by_method["time_pf"].to_list(),
                color=_color,
                marker=_markerstyle,
                linestyle=_linestyle,
                linewidth=0.5,
                markersize=marker_size,
                label=method_name)

    ax1[0].set_xscale('log')
    ax1[0].set_yscale('log')
    ax1[0].grid(which="minor", linewidth=linewidth_minor_grid, dashes=dashes_minor_grid)
    ax1[0].grid(which="major", linestyle="-", linewidth=linewidth_major_grid)
    ax1[0].set_xticks(time_steps_list)
    ax1[0].tick_params(axis='both', which='major', labelsize=6)
    ax1[0].set_ylabel("Time power flow\n[sec]", fontsize=6)
    ax1[0].set_xlabel("Dimensional tensor elements [" + r"$\tau$"+ "]", fontsize=6)
    ax1[0].set_title("(c)", fontsize=6)

    ax1[0].legend(["SAM", "NR (Sparse)", "BFS", "Tensor (Dense)", "Tensor (Sparse)", "Tensor (GPU)"],
                  fontsize=6,
                  ncol=3,
                  bbox_to_anchor=(2.6, -0.35),
                  handlelength=1.5
                  )

    # Plot of time sweep:
    # ax1[0].xaxis.set_ticks(time_sweep_major_xticks, minor=False)

    ax1[0].xaxis.set_ticks(np.logspace(np.log10(1), np.log10(1e5), 6), minor=False)
    ax1[0].yaxis.set_ticks(time_sweep_major_yticks, minor=False)

    ax1[0].xaxis.set_ticks(time_sweep_minor_xticks, minor=True)
    ax1[0].yaxis.set_ticks(time_sweep_minor_yticks, minor=True)
    ax1[0].set_xlim(time_sweep_xlim)
    ax1[0].set_ylim(time_sweep_ylim)

    # ==================================================================================================================
    # Fourth subplot (Second row, second column)
    # ==================================================================================================================
    # Reset colors
    colors = cycle(["red", "blue", "green", "olive", "purple", "fuchsia", "peru"])
    line_styles = cycle(["-", "--", "-.", ":", "-", "--"])
    marker_styles = cycle([".", ",", "o", "v", "<", ">", "1"])

    algorithm_names = fitting_parameters_frame["algorithm"].unique()
    ax2 = ax1[1].twinx()
    for algorithm_name, color, _marker_style in zip(method_names, colors, marker_styles):
        print(_marker_style)
        filtered_frame_by_algorithm = fitting_parameters_frame.loc[
            fitting_parameters_frame["algorithm"] == algorithm_name]
        ax2.plot(filtered_frame_by_algorithm["grid_size"].values.astype(float),
                 filtered_frame_by_algorithm["c"].values,
                 linestyle="-",
                 color=color,
                 marker=_markerstyle,
                 markersize=marker_size,
                 label="'c' " + algorithm_name)
        ax1[1].plot(filtered_frame_by_algorithm["grid_size"].values.astype(float),
                 filtered_frame_by_algorithm["k"].values,
                 linestyle="--",
                 color=color,
                 marker=_markerstyle,
                 markersize=marker_size,
                 label="'k' " + algorithm_name)

    ax1[1].grid(which="minor", linewidth=linewidth_minor_grid, dashes=dashes_minor_grid)
    ax1[1].grid(which="major", linestyle="-", linewidth=linewidth_major_grid)

    ax2.set_ylabel(r"$c$", fontsize=6)
    ax1[1].set_ylabel(r"$k$", fontsize=6)

    ax1[1].set_ylim((0, 1.5))
    ax2.set_ylim((0, 1.5))

    ax1[1].set_title("(d)", fontsize=6)

    # Fromula t_c = c * (n ^ k)
    # ax.legend(fontsize="x-small", loc="upper center")
    # ax1[1].legend(fontsize="x-small", loc="center right")
    # ax.grid(which="major", linestyle="--", linewidth=0.8)
    # ax2.grid(which="major", linestyle="-", linewidth=0.9)
    ax1[1].set_xlabel("Grid size nodes [" + r"$n$" + "]", fontsize=6)
    plt.savefig(r"figures\experiments.pdf")







