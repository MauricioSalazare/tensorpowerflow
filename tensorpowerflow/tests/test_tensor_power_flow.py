from tensorpowerflow import GridTensor
import numpy as np

def test_check_output_voltage_array_dimensions():
    network = GridTensor()
    active_ns = np.random.normal(50 ,
                                 scale=10,
                                 size=(10, 20, 33)).round(3) # Assume 1 slack variable
    reactive_ns = (active_ns * .1).round(3)

    solution = network.run_pf(active_power=active_ns, reactive_power=reactive_ns)

    assert solution["v"].shape == active_ns.shape

if __name__ == "__main__":
    test_check_output_voltage_array_dimensions()