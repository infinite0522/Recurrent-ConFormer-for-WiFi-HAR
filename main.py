import json
from trainer import train

def main(config_file):
    args = load_json(config_file)
    train(args)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


if __name__ == '__main__':
    main("configs/UT-HAR.json")
