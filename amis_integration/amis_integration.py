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

cache: dict[(float, float), pandas.DataFrame] = {}

def returnYearlyPrevalenceEstimate(
    coverageFileName, coverageTextFileStorageName, paramFileName, demogName, R0, k, N
):
    if (R0, k) in cache:
        return cache[(R0, k)]
    print(f"starting run for {R0} {k}")
    cov = parse_coverage_input(coverageFileName, coverageTextFileStorageName)
    # initialize the parameters
    params = sch_simulation.helsim_RUN_KK.loadParameters(paramFileName, demogName)
    # add coverage data to parameters file
    params = readCoverageFile(coverageTextFileStorageName, params)
    # add vector control data to parameters
    params = parse_vector_control_input(coverageFileName, params)

    # update R0, k and population size as desired
    params.R0 = R0
    params.k = k
    params.N = N

    # setup the initial simData

    simData = setupSD(params)

    # cset the survey type to Kato Katz with duplicate slide
    surveyType = "KK2"

    # the following number dictates the number of events (e.g. worm deaths) we allow to happen before updating other parts of the model
    # the higher this number the faster the simulation (though there is a check so that there can't be too many events at once)
    # the higher this number the greater the potential for errors in the model accruing.
    # 5 is a reasonable level of compromise for speed and errors, but using a lower value such as 3 is also quite good
    mult = 5
    # run a single realization
    results, SD = sch_simulation.helsim_RUN_KK.doRealizationSurveyCoveragePickle(
        params, surveyType, simData, mult
    )

    results = [results]
    # process the output
    output = extractHostData(results)

    # do a single simulation
    numReps = 1
    PrevalenceEstimate = getPrevalenceWholePop(
        output, params, numReps, params.Unfertilized, "KK2", 1
    )
    cache[(R0, k)] = PrevalenceEstimate
    return PrevalenceEstimate


def extract_relevant_results(results: pandas.DataFrame) -> float:
    num_years = len(results)
    # TODO: for now we assume we are interested in fitting the last year
    return results["draw_1"][num_years - 1]


def run_model_with_parameters(seeds, parameters):
    if len(seeds) != len(parameters):
        raise ValueError(
            f"Must have same number of seeds as parameters {len(seeds)} != {len(parameters)}"
        )
    print(parameters)
    num_runs = len(seeds)

    coverageFileName = "mansoni_coverage_scenario_0.xlsx"  # no intervention
    # coverageFileName = 'mansoni_coverage_scenario_1.xlsx' # annual MDA with original drug and ~65% coverage
    # coverageFileName = 'mansoni_coverage_scenario_3a_1.xlsx' # higher coverage, improved drug and vaccination

    # which demography to use
    demogName = "UgandaRural"
    # file name to store coverage information in
    coverageTextFileStorageName = "Man_MDA_vacc.txt"
    # standard parameter file path (in sch_simulation/data folder)
    paramFileName = "mansoni_params.txt"

    final_prevalence_for_each_run = []

    for seed, parameter_set in zip(seeds, parameters):

        # change parameters here to examine the impact of R0 and k.
        R0 = parameter_set[0]
        k = parameter_set[1]

        # the higher the value of N, the more consistent the results will be
        # though the longer the simulation will take
        N = 100

        results = returnYearlyPrevalenceEstimate(
            coverageFileName,
            coverageTextFileStorageName,
            paramFileName,
            demogName,
            R0,
            k,
            N,
        )

        prevalence = extract_relevant_results(results)
        final_prevalence_for_each_run.append(prevalence)

    # TODO: Extract the relevant prevalence info from the results
    print(final_prevalence_for_each_run)
    results_np_array = np.array(final_prevalence_for_each_run).reshape(500, 1)
    print(results_np_array.shape)
    return results_np_array
