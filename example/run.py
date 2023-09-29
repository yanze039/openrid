import sys
import argparse

# add the path to the openrid package
sys.path.append("..")
from openrid import ReinforcedDynamics

def parser_config():
    parser = argparse.ArgumentParser(description="Run Reinforced Dynamics")
    parser.add_argument("CONFIG", type=str, default="rid.yaml", help="path to the config file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parser_config()
    rid = ReinforcedDynamics(config=args.CONFIG)
    rid.run()