import sys

# add the path to the openrid package
sys.path.append("..")
from openrid import ReinforcedDynamics

if __name__ == "__main__":
    rid = ReinforcedDynamics(config="./ala2.yaml")
    rid.run()