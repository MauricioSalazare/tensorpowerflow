from tensorpowerflow import generate_network

aa = generate_network(nodes=100, child=3, plot_graph=True, load_factor=2, line_factor=3)