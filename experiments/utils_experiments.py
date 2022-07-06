import numpy as np
from tqdm import trange

def run_test(pf_method, active_power_data, reactive_power_data, method_name, single_step=False):
    N_TS = active_power_data.shape[0]
    # print("=" * 72)
    # print(f"Build compilers. N_Power flows: {N_TS}: Method: {method_name}")
    # if single_step:
    #     _ = pf_method()
    # else:
    #     _ = pf_method(active_power=active_power_data, reactive_power=reactive_power_data)

    print("*" * 72)
    print("Real performance measurements:")
    if single_step:
        voltage_solutions = []
        times_pf = []
        times_pre_pf = []
        times_total = []
        iterations = []
        for i in trange(N_TS, desc=f'Method: {method_name}'):
            solution = pf_method(active_power=active_power_data[i, :],
                                 reactive_power=reactive_power_data[i, :])
            voltage_solutions.append(solution["v"])
            times_pf.append(solution["time_pf"])
            times_pre_pf.append(solution["time_pre_pf"])
            times_total.append(solution["time_algorithm"])
            iterations.append(solution["iterations"])

        solution_method = {
                           "v": np.array(voltage_solutions),
                           "time_pre_pf": np.sum(times_pre_pf),
                           "time_pf": np.sum(times_pf),
                           "time_algorithm": np.sum(times_total),
                           "iterations": np.mean(iterations),
                           "convergence": np.nan
                           }
    else:
        solution = pf_method(active_power=active_power_data,
                             reactive_power=reactive_power_data)
        solution_method = solution

    test_results = {method_name: solution_method}
    test_results[method_name]["time_steps"] = N_TS

    return test_results