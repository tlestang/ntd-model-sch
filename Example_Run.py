from sch_simulation.helsim_RUN_KK import *
from sch_simulation.helsim_FUNC_KK import *
from sch_simulation.helsim_FUNC_KK.configuration import getEquilibrium
import matplotlib.pyplot as plt

# which set of intervetions should we use
coverageFileName = 'mansoni_coverage_scenario_0.xlsx' # no intervention
#coverageFileName = 'mansoni_coverage_scenario_1.xlsx' # annual MDA with original drug and ~65% coverage
#coverageFileName = 'mansoni_coverage_scenario_3a_1.xlsx' # higher coverage, improved drug and vaccination

# which demography to use
demogName = 'UgandaRural'
# file name to store coverage information in 
coverageTextFileStorageName = 'Man_MDA_vacc.txt'
# standard parameter file path (in sch_simulation/data folder)
paramFileName = 'mansoni_params.txt'

# change parameters here to examine the impact of R0 and k. 
R0 = 3
k = 0.04

# the higher the value of N, the more consistent the results will be 
# though the longer the simulation will take
N = 1000

# the following number dictates the number of events (e.g. worm deaths) we allow to happen before updating other parts of the model
# the higher this number the faster the simulation (though there is a check so that there can't be too many events at once)
# the higher this number the greater the potential for errors in the model accruing.
# 5 is a reasonable level of compromise for speed and errors, but using a lower value such as 3 is also quite good
mult = 5

def returnYearlyPrevalenceEstimate(coverageFileName, 
                                   coverageTextFileStorageName, 
                                   paramFileName, 
                                   demogName,
                                   R0,
                                   k,
                                   N):
    cov = parse_coverage_input(coverageFileName,
     coverageTextFileStorageName)
    # initialize the parameters
    params = loadParameters(paramFileName, demogName)
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

    # run a single realization
    results, SD = doRealizationSurveyCoveragePickle(params, surveyType, simData, mult)

    results = [results]
    # process the output
    output = extractHostData(results)
    
    # do a single simulation
    numReps = 1
    PrevalenceEstimate = getPrevalenceWholePop(output, params, numReps, params.Unfertilized,  'KK2', 1)

    return PrevalenceEstimate



PrevalenceEstimate = returnYearlyPrevalenceEstimate(coverageFileName, 
                                                    coverageTextFileStorageName, 
                                                    paramFileName, 
                                                    demogName,
                                                    R0,
                                                    k,
                                                    N)

print(PrevalenceEstimate)


plt.plot(PrevalenceEstimate.draw_1)
plt.title("Prevalence estimates")
plt.xlabel("Time")
plt.legend()
plt.show()
