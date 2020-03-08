import argparse
import json

ARGUMENTS = [
    "window_size",
    "learning_rate",
    "batch_size",
    "rolling_training_steps",
    "weight_decay_dense",
    "weight_decay_output",
]
CONFIG_FILE_DIR = "../pgportfolio/net_config.json"


def parse_args():
    parser = argparse.ArgumentParser()
    for argument in ARGUMENTS:
        parser.add_argument(f"--{argument}", dest=argument, type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    with open(CONFIG_FILE_DIR, "r") as config_file:
        config = json.load(config_file)
        if args.weight_decay_dense:
            config["layers"][1]["weight_decay"] = args.weight_decay_dense
        if args.weight_decay_output:
            config["layers"][2]["weight_decay"] = args.weight_decay_output
        if args.learning_rate:
            config["training"]["learning_rate"] = args.learning_rate
            config["trading"]["learning_rate"] = args.learning_rate
        if args.batch_size:
            config["training"]["batch_size"] = args.batch_size
        if args.window_size:
            config["input"]["window_size"] = args.window_size
        if args.rolling_training_steps:
            config["trading"]["rolling_training_steps"] = args.rolling_training_steps

    with open(CONFIG_FILE_DIR, "w") as outfile:
        json.dump(config, outfile, indent=4)


if __name__ == "__main__":
    main()
