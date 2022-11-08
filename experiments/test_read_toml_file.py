import toml

print(toml.load("config_example.toml"))
aa = toml.load("config_example.toml")
print(toml.dumps(aa))
aa["default_inventory"]["warrior"]

print(toml.load("config.toml"))
bb = toml.load("config.toml")
print(toml.dumps(bb))
bb["params"]["contact"]