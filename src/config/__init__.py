import yaml
import os

# Open the YAML configuration file
script_dit= os.path.dirname(__file__)
yaml_file = os.path.join(script_dit, 'config.yaml')
with open(yaml_file , 'r') as file:
    cf = yaml.safe_load(file)