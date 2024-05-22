from dataclasses import dataclass
from functools import cache
from typing import Literal
import pandas
import sch_simulation
import numpy as np
from sch_simulation.helsim_FUNC_KK.configuration import setupSD
from sch_simulation.helsim_FUNC_KK.file_parsing import (
    parse_coverage_input,
    parse_vector_control_input,
    readCoverageFile,
)
from sch_simulation.helsim_FUNC_KK.results_processing import (
    extractHostData,
    getPrevalenceWholePop,
)
import sch_simulation.helsim_RUN_KK

ParameterSet = tuple[float, float]


@dataclass(eq=True, frozen=True)
class FixedParameters:
    number_hosts: int
    coverage_file_name: str
    # which demography to use
    demography_name: str
    survey_type: Literal["KK1", "KK2", "POC-CCA", "PCR"]
    # file name to store coverage information in
    coverage_text_file_storage_name: str
    # standard parameter file path (in sch_simulation/data folder)
    parameter_file_name: str


@cache
def returnYearlyPrevalenceEstimate(R0, k, fixed_parameters: FixedParameters):
    print(f"starting run for {R0} {k}")
    cov = parse_coverage_input(
        fixed_parameters.coverage_file_name,
        fixed_parameters.coverage_text_file_storage_name,
    )
    # initialize the parameters
    params = sch_simulation.helsim_RUN_KK.loadParameters(
        fixed_parameters.parameter_file_name, fixed_parameters.demography_name
    )
    # add coverage data to parameters file
    params = readCoverageFile(fixed_parameters.coverage_text_file_storage_name, params)
    # add vector control data to parameters
    params = parse_vector_control_input(fixed_parameters.coverage_file_name, params)

    # update R0, k and population size as desired
    params.R0 = R0
    params.k = k
    params.N = fixed_parameters.number_hosts

    # setup the initial simData

    simData = setupSD(params)

    # the following number dictates the number of events (e.g. worm deaths) we allow to happen before updating other parts of the model
    # the higher this number the faster the simulation (though there is a check so that there can't be too many events at once)
    # the higher this number the greater the potential for errors in the model accruing.
    # 5 is a reasonable level of compromise for speed and errors, but using a lower value such as 3 is also quite good
    mult = 5
    # run a single realization
    results, SD = sch_simulation.helsim_RUN_KK.doRealizationSurveyCoveragePickle(
        params, fixed_parameters.survey_type, simData, mult
    )

    results = [results]
    # process the output
    output = extractHostData(results)

    # do a single simulation
    numReps = 1
    PrevalenceEstimate = getPrevalenceWholePop(
        output, params, numReps, params.Unfertilized, fixed_parameters.survey_type, 1
    )
    return PrevalenceEstimate


def extract_relevant_results(results: pandas.DataFrame) -> float:
    num_years = len(results)
    # TODO: for now we assume we are interested in fitting the last year
    return results["draw_1"][num_years - 1]


def run_model_with_parameters(seeds, parameters, fixed_parameters: FixedParameters):
    if len(seeds) != len(parameters):
        raise ValueError(
            f"Must have same number of seeds as parameters {len(seeds)} != {len(parameters)}"
        )

    final_prevalence_for_each_run = []

    for seed, parameter_set in zip(seeds, parameters):
        R0 = parameter_set[0]
        k = parameter_set[1]

        results = returnYearlyPrevalenceEstimate(R0, k, fixed_parameters)

        prevalence = extract_relevant_results(results)
        final_prevalence_for_each_run.append(prevalence)

    results_np_array = np.array(final_prevalence_for_each_run).reshape(500, 1)
    return results_np_array
