from milano.backends import SLURMBackend
from milano.search_algorithms import RandomSearch
import os
import pathlib

# specify path to the script that is going to be tuned
# path has to be absolute or relative to tune.py script
# this should be on a driver machine
script_to_run = os.path.join(
    pathlib.Path(__file__).parent.absolute(), "experiment_run.sh"
)

# specify the tunable parameters as cmd arguments and their possible ranges
params_to_tune = {
    "--window_size": {"type": "values", "values": [1, 5, 10, 20, 30, 50]},
    "--learning_rate": {"type": "values", "values": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]},
    "--batch_size": {"type": "values", "values": [1, 10, 20, 50, 100]},
    "--weight_decay_dense": {
        "type": "values",
        "values": [0, 5e-10, 5e-9, 5e-8, 5e-7, 5e-6],
    },
    "--weight_decay_output": {
        "type": "values",
        "values": [0, 5e-9, 5e-8, 5e-7, 5e-6, 5e-5],
    },
}

# specify result pattern used to parse logs
result_pattern = "the final portfolio value is "

# maximize or minimize
objective = "maximize"

num_evals = 1000
num_workers = min(100, num_evals)

# BACKEND parameters. We will use SLURMBackend to run on DB Cluster
backend = SLURMBackend
backend_params = {
    "workers_config": {
        "num_workers": num_workers,  # NUMBER OF SLURM *NODES* to run at a time.
        "partition": "Teach-Short",  # PARTITION
        "username": "s1545572",  # CHANGE THIS
        "key_path": None,  # CHANGE THIS
        "entrypoint": "mlp",  # CHANGE THIS
    },
}

search_algorithm = RandomSearch
search_algorithm_params = {
    "num_evals": num_evals,
}
