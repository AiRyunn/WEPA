import logging
import os
from pathlib import Path

import papermill as pm

logging.basicConfig(level='INFO', format="%(message)s")

os.makedirs('./experiments/output', exist_ok=True)
experiment = Path(__file__).stem.replace("run_", "")

parameters_list = [
    {
        "device": "0",
        'model_name': 'meta-llama/Llama-3.2-3B',
        'corruption_type': "substitution",
        'dataset_size': 200,
    },
    {
        "device": "1",
        'model_name': 'meta-llama/Llama-3.2-3B',
        'corruption_type': "deletion",
        'dataset_size': 200,
    },
    {
        "device": "2",
        'model_name': 'meta-llama/Llama-3.2-3B',
        'corruption_type': "insertion",
        'dataset_size': 200,
    },
    {
        "device": "3",
        'model_name': 'mistralai/Mistral-7B-v0.3',
        'corruption_type': "substitution",
        'dataset_size': 200,
    },
    {
        "device": "4",
        'model_name': 'mistralai/Mistral-7B-v0.3',
        'corruption_type': "deletion",
        'dataset_size': 200,
    },
    {
        "device": "5",
        'model_name': 'mistralai/Mistral-7B-v0.3',
        'corruption_type': "insertion",
        'dataset_size': 200,
    },
]


for parameters in parameters_list:
    output_path = f'./experiments/output/{experiment}/{parameters["corruption_type"]}/{parameters["model_name"]}.ipynb'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pm.execute_notebook(
        f'./experiments/{experiment}.ipynb',
        output_path,
        parameters=parameters,
        cwd='./experiments',
        log_output=True,
    )
