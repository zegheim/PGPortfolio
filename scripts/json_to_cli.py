import argparse
import json

ARGUMENTS = [
    "window_size",
    "learning_rate",
    "batch_size",
    "weight_decay_dense",
    "weight_decay_output",
]
CONFIG_FILE_DIR = "pgportfolio/net_config.json"
STEPS_PER_BATCH_SIZE = 3000


def parse_args():
    parser = argparse.ArgumentParser()
    for argument in ARGUMENTS:
        parser.add_argument(f"--{argument}", dest=argument)

    return parser.parse_args()


def main():
    args = parse_args()
    with open(CONFIG_FILE_DIR, "r") as config_file:
        config = json.load(config_file)
        if args.weight_decay_dense:
            config["layers"][1]["weight_decay"] = float(args.weight_decay_dense)
        if args.weight_decay_output:
            config["layers"][2]["weight_decay"] = float(args.weight_decay_output)
        if args.learning_rate:
            config["training"]["learning_rate"] = float(args.weight_decay_output)
            config["trading"]["learning_rate"] = float(args.weight_decay_output)
        if args.batch_size:
            batch_size = int(args.batch_size)
            config["training"]["batch_size"] = batch_size
            config["training"]["steps"] = batch_size * STEPS_PER_BATCH_SIZE
        if args.window_size:
            config["input"]["window_size"] = int(args.batch_size)

    with open(CONFIG_FILE_DIR, "w") as outfile:
        json.dump(config, outfile, indent=4)


if __name__ == "__main__":
    main()
