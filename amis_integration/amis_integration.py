import numpy as np

def run_model_with_parameters(seeds, parameters):
    if len(seeds) != len(parameters):
        raise ValueError(f'Must have same number of seeds as parameters {len(seeds)} != {len(parameters)}')
    print(parameters)
    num_runs = len(seeds)
        
    # TODO: Extract the relevant prevalence info from the results
    return np.full((num_runs, 1), 0.5)
