import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
import pandas as pd
import pkg_resources

try:
    import pandapower as pp
except ModuleNotFoundError:
    pass

def _load_default_34_node_case():
    stream_node_data = pkg_resources.resource_stream(__name__, "data/Lines_34.csv")
    stream_line_data = pkg_resources.resource_stream(__name__, "data/Nodes_34.csv")

    nodes_frame = pd.read_csv(stream_node_data, encoding='utf-8')
    lines_frame = pd.read_csv(stream_line_data, encoding='utf-8')

    return nodes_frame, lines_frame

def _load_default_2_node_case():
    stream_node_data = pkg_resources.resource_stream(__name__, "data/Lines_2.csv")
    stream_line_data = pkg_resources.resource_stream(__name__, "data/Nodes_2.csv")

    nodes_frame = pd.read_csv(stream_node_data, encoding='utf-8')
    lines_frame = pd.read_csv(stream_line_data, encoding='utf-8')

    return nodes_frame, lines_frame

def generate_network(nodes, child=3, plot_graph=False, load_factor=2, line_factor=3):
    LINES = nodes - 1
    G = nx.full_rary_tree(child, nodes)

    if plot_graph:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        nx.draw_kamada_kawai(G, node_size=100, with_labels=True, font_size='medium', ax=ax)

    assert nodes == len(G.nodes)
    assert LINES == len(G.edges)

    # Generate a pandas dataframe
    PCT, ICT, ZCT = 1, 0, 0
    Tb, Pct, Ict, Zct = 0, PCT, ICT, ZCT
    nodes_ = pd.DataFrame(list(G.nodes), columns=["NODES"]) + 1
    power = pd.DataFrame({"PD": np.random.normal(300 / load_factor, scale=50, size=nodes).round(2),
                          "QD": np.random.normal(100 / load_factor, scale=50, size=nodes).round(2)})
    nodes_properties_ = pd.DataFrame(np.tile([[Tb, Pct, Ict, Zct]], (nodes, 1)),
                                     columns=["Tb", "Pct", "Ict", "Zct"])
    nodes_properties = pd.concat([power, nodes_properties_], axis=1)
    nodes_properties = nodes_properties.astype(
        {"Tb": int, "PD": float, "QD": float, "Pct": int, "Ict": int, "Zct": int})
    nodes_properties = nodes_properties[["Tb", "PD", "QD", "Pct", "Ict", "Zct"]]
    nodes_properties.loc[0] = 1, 0.0, 0.0, PCT, ICT, ZCT  # Slack
    nodes_frame = pd.concat([nodes_, nodes_properties], axis=1)

    # R, X = 0.3144, 0.054
    R, X = 0.3144 / line_factor, 0.054 / line_factor
    lines = pd.DataFrame.from_records(list(G.edges), columns=["FROM", "TO"]) + 1  # Count starts from 1
    lines_properties = pd.DataFrame(np.tile([[R, X, 0, 1, 1]], (LINES, 1)),
                                    columns=["R", "X", "B", "STATUS", "TAP"])
    lines_properties = lines_properties.astype({"R": float, "X": float, "B": int, "STATUS": int, "TAP": int})
    lines_frame = pd.concat([lines, lines_properties], axis=1)

    return nodes_frame, lines_frame


def create_pandapower_net(branch_info_: pd.DataFrame, bus_info_: pd.DataFrame):
    branch_info = branch_info_.copy()
    bus_info = bus_info_.copy()

    start = perf_counter()
    net = pp.create_empty_network()
    # Add buses
    bus_dict = {}
    for i, (idx, bus_name) in enumerate(bus_info["NODES"].iteritems()):
        bus_dict[bus_name] = pp.create_bus(net, vn_kv=11., name=f"Bus {bus_name}")

    # Slack
    bus_slack = bus_info[bus_info["Tb"] == 1]["NODES"].values
    assert len(bus_slack.shape) == 1 and bus_slack.shape[0] == 1, "Only one slack bus supported"
    pp.create_ext_grid(net, bus=bus_dict[bus_slack.item()], vm_pu=1.00, name="Grid Connection")

    # Lines
    for i, (idx, (from_bus, to_bus, res, x_react, b_susceptance)) in enumerate(
            branch_info[["FROM", "TO", "R", "X", "B"]].iterrows()):
        pp.create_line_from_parameters(net,
                                       from_bus=bus_dict[from_bus], to_bus=bus_dict[to_bus],
                                       length_km=1, r_ohm_per_km=res, x_ohm_per_km=x_react, c_nf_per_km=b_susceptance,
                                       max_i_ka=10, name=f"Line {i + 1}")

    # Loads:
    for i, (idx, (node, p_kw, q_kvar)) in enumerate(bus_info[["NODES", "PD", "QD"]].iterrows()):
        pp.create_load(net, bus=bus_dict[node], p_mw=p_kw / 1000., q_mvar=q_kvar / 1000., name=f"Load")
    print(f"Create net time: {perf_counter() - start}")

    return net

def net_test(net):
    """Compute the voltage for the base case. This only works for the net of 34 buses"""

    v_solution = [0.98965162 + 0.00180549j, 0.98060256 + 0.00337785j, 0.96828145 + 0.00704551j,
                  0.95767051 + 0.01019764j, 0.94765203 + 0.01316654j, 0.94090964 + 0.01600068j,
                  0.93719984 + 0.01754998j, 0.93283877 + 0.01937559j, 0.93073823 + 0.02026054j,
                  0.9299309 + 0.02058985j, 0.92968994 + 0.02068728j, 0.98003142 + 0.00362498j,
                  0.97950885 + 0.00385019j, 0.97936712 + 0.00391065j, 0.97935604 + 0.0039148j,
                  0.93971131 + 0.01547898j, 0.93309482 + 0.01739656j, 0.92577912 + 0.01988823j,
                  0.91988489 + 0.02188907j, 0.91475251 + 0.02362566j, 0.90888169 + 0.02596304j,
                  0.90404908 + 0.02788248j, 0.89950353 + 0.02968449j, 0.89731375 + 0.03055177j,
                  0.89647201 + 0.03088507j, 0.89622055 + 0.03098473j, 0.94032081 + 0.01625577j,
                  0.93992817 + 0.01642583j, 0.93973182 + 0.01651086j, 0.9301316 + 0.02052908j,
                  0.92952481 + 0.02079761j, 0.92922137 + 0.02093188j, 0.92912022 + 0.02097663j]
    v_solution = np.array(v_solution, dtype="complex128")

    for pf_algorithm in ["nr", "bfsw"]:
        print(f"Testing: {pf_algorithm} - Algorithm")
        start = perf_counter()
        if pf_algorithm == "bfsw":
            pp.runpp(net, algorithm=pf_algorithm, numba=False, v_debug=True, VERBOSE=False, tolerance_mva=1e-6)
            print(f"BFSW. Iterations: {net._ppc['iterations']}. PF time: {net._ppc['et']}")
        elif pf_algorithm == "nr":
            pp.runpp(net, algorithm=pf_algorithm, numba=False, v_debug=True, VERBOSE=False, tolerance_mva=1e-6)
            print(f"NR. Iterations: {net._ppc['iterations']}. PF time: {net._ppc['et']}")
        print(f"Total pf time: {perf_counter() - start}.")

        v_real = net.res_bus["vm_pu"].values * np.cos(np.deg2rad(net.res_bus["va_degree"].values))
        v_img = net.res_bus["vm_pu"].values * np.sin(np.deg2rad(net.res_bus["va_degree"].values))
        v_result = v_real + 1j * v_img
        assert np.allclose(v_result[1:], v_solution)
        print("Test OK.")

    return True

if __name__ == "__main__":
    _load_default_34_node_case()

