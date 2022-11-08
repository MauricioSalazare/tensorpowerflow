from utils_experiments import build_valid_network, save_grid_to_file
from tqdm import tqdm
from time import perf_counter

"""
Creates all the the solvable cases for different grid sizes.
The files are saved in .zip files, which inside are parquet files (to increase the speed of loading and reduce disk
space).
"""

def create_case(n_nodes, n_power_flows):
    network_data = build_valid_network(n_nodes=n_nodes,
                                       n_power_flows=n_power_flows,
                                       lambda_f=0.5,
                                       truncate_to=100,
                                       seed=12345)

    save_grid_to_file(file_name=f"case_nodes_{n_nodes}",  # Does not require the extension
                      line_data=network_data["network"].branch_info,
                      node_data=network_data["network"].bus_info,
                      active_power=network_data["active_power"],
                      reactive_power=network_data["reactive_power"],
                      voltage_solutions=network_data["solutions"]["v"])


if __name__ == "__main__":
    start = perf_counter()
    n_nodes_list = [10, 30, 100, 200, 300, 500, 1000, 3_000, 5_000]
    for n_nodes in tqdm(n_nodes_list):
        create_case(n_nodes=n_nodes, n_power_flows=100_000)
    end = perf_counter()
    print(f"Total time scenario creation: {(end - start):.3f} sec.")