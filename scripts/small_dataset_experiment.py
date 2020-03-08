from milano.backends import SLURMBackend
from milano.search_algorithms import RandomSearch
import os
import pathlib

# specify path to the script that is going to be tuned
# path has to be absolute or relative to tune.py script
# this should be on a driver machine
script_to_run = os.path.join(pathlib.Path(__file__).parent.absolute(), "embeddings_experiment_run.sh")

# specify the tunable parameters as cmd arguments and their possible ranges
params_to_tune = {
    "--pad_size": {
        "type": "values", "values": [2, 5, 10, 20, 40, 80, 160]
    },
    "--tag_threshold": {
        "type": "values", "values": [2, 5, 10, 20, 40, 80]
    },
    "--layer_capacity": {
        "type": "values", "values": [64, 128]
    },
    "--pooling_layer": {
        "type": "values", "values": ["GlobalAveragePooling", "GlobalMaxPool"]
    },
    "--batch_size": {
        "type": "values", "values": [128, 256]
    },
    "--learning_rate": {
        "type": "values", "values": [0.1, 0.01, 0.001]
    },
    "--dropout_rate": {
        "type": "values", "values": [0, 0.1, 0.2, 0.3, 0.4]
    },
    "--l2_reg_factor": {
        "type": "values", "values": [0, 0.1, 0.01, 0.001, 0.0001]
    },
}

# specify result pattern used to parse logs
result_pattern = "Validation MAP is: "

# maximize or minimize
objective = "maximize"

num_evals = 400
num_workers = min(100, num_evals)

# BACKEND parameters. We will use SLURMBackend to run on DB Cluster
backend = SLURMBackend
backend_params = {
    "workers_config": {
        "num_workers": num_workers,  # NUMBER OF SLURM *NODES* to run at a time.
        "partition": "Teach-Standard",  # PARTITION
        "username": "s1614973",  # CHANGE THIS
        "key_path": None,  # CHANGE THIS
        "entrypoint": "mlp",  # CHANGE THIS
    },
}

search_algorithm = RandomSearch
search_algorithm_params = {
    "num_evals": num_evals,
}
