import copy
from dataclasses import dataclass
import os
import pickle
import pandas
import pkg_resources
import sch_simulation
import numpy as np
import sch_simulation.helsim_RUN_KK
from sch_simulation.helsim_FUNC_KK.helsim_structures import Parameters, SDEquilibrium

ParameterSet = tuple[float, float]

def load_simulation_data(simulated_data_path: str) -> SDEquilibrium:
    full_path = pkg_resources.resource_filename('sch_simulation', os.path.join('data', simulated_data_path))
    with open(full_path, 'rb') as simulated_data_pickle_file:
        simulated_data = pickle.load(simulated_data_pickle_file)
        if isinstance(simulated_data, dict):
            # TODO: this pickle file is not an up to date simulated data
            print(simulated_data)
            return SDEquilibrium(**simulated_data, id = 0, n_treatments = 0, n_treatments_population = 1, n_surveys = 0, n_surveys_population = 1)
        raise ValueError(f'Pickle file for simulated data not a list of simulated data: {type(simulated_data)}')

def adapt_parameters_to_shape(parameters:tuple[list[float], list[float]]) -> pandas.DataFrame:
    '''The algorithm expects a dataframe where each row is a R0/k pair
    The R interface for AMIS provides these pairs as a '''
    return pandas.DataFrame(data={'R0': parameters[0], 'k': parameters[1]})


def run_model_with_parameters(seeds, parameters):
    if len(seeds) != len(parameters):
        raise ValueError(f'Must have same number of seeds as parameters {len(seeds)} != {len(parameters)}')
    print(parameters)
    num_runs = len(seeds)
    
    # TODO: this probably should be doing the burnin for each paramater set?
    # TODO: probably shouldn't be using the same sim data for each run
    simulation_data = [load_simulation_data('SD1.pickle')] * num_runs

    # TODO: this checks nothing as we duplicate to the required amount
    if len(simulation_data) < num_runs:
        raise ValueError('Need starting point simulation data for at least each of the runs')
    
    base_parameters = sch_simulation.helsim_RUN_KK.loadParameters(paramFileName='sch_example.txt', demogName='Default')

    indices = range(num_runs)
    sim_params = adapt_parameters_to_shape(parameters)

    results = map(lambda run_index: sch_simulation.helsim_RUN_KK.multiple_simulations(
            copy.deepcopy(base_parameters), 
            copy.deepcopy(simulation_data), 
            sim_params,
            indices,
            run_index
        ), indices)
    
    print(results)
        
    # TODO: Extract the relevant prevalence info from the results
    return np.full((num_runs, 1), 0.5)
