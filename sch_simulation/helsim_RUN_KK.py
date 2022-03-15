from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import numpy as np
import copy

from sch_simulation.helsim_FUNC_KK import *
num_cores = multiprocessing.cpu_count()


def loadParameters(paramFileName, demogName):
    '''
    This function loads all the parameters from the input text
    files and organizes them in a dictionary.

    Parameters
    ----------
    paramFileName: str
        name of the input text file with the model parameters;

    demogName: str
        subset of demography parameters to be extracted;

    Returns
    -------
    params: dict
        dictionary containing the parameter names and values;
    '''

    # load the parameters
    params = readParams(paramFileName=paramFileName, demogName=demogName)

    # configure the parameters
    params = configure(params)

    # update the parameters
    params['psi'] = getPsi(params)
    params['equiData'] = getEquilibrium(params)

    return params


def doRealization(params, i):
    '''
    This function generates a single simulation path.

    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;

    i: int
        iteration number;

    Returns
    -------
    results: list
        list with simulation results;
    '''

    # setup simulation data
    simData = setupSD(params)

    # start time
    t = 0

    # end time
    maxTime = copy.deepcopy(params['maxTime'])

    # time at which to update the freelive population
    freeliveTime = t

    # times at which data should be recorded
    outTimes = copy.deepcopy(params['outTimings'])

    # time when data should be recorded next
    nextOutIndex = np.argmin(outTimes)
    nextOutTime = outTimes[nextOutIndex]

    # time at which individuals' age is advanced next
    ageingInt = 1 / 52
    nextAgeTime = 1 / 52
    maxStep = 1 / 52

    # time at which individuals receive next chemotherapy
    currentchemoTiming1 = copy.deepcopy(params['chemoTimings1'])
    currentchemoTiming2 = copy.deepcopy(params['chemoTimings2'])

    currentVaccineTimings = copy.deepcopy(params['VaccineTimings'])

    nextChemoIndex1 = np.argmin(currentchemoTiming1)
    nextChemoIndex2 = np.argmin(currentchemoTiming2)
    nextVaccineIndex = np.argmin(currentVaccineTimings)

    nextChemoTime1 = currentchemoTiming1[nextChemoIndex1]
    nextChemoTime2 = currentchemoTiming2[nextChemoIndex2]
    nextVaccineTime = currentVaccineTimings[nextVaccineIndex]
    # next event
    nextStep = np.min([nextOutTime, t + maxStep, nextChemoTime1,
                      nextChemoTime2, nextAgeTime, nextVaccineTime])

    results = []  # initialise empty list to store results

    # run stochastic algorithm
    while t < maxTime:
        if t%1 == 1:
            print(t)
        rates = calcRates2(params, simData)
        sumRates = np.sum(rates)

        # if the rate is such that nothing's likely to happen in the next 10,000 years,
        # just fix the next time step to 10,000
        if sumRates < 1e-4:

            dt = 10000

        else:


            dt = np.random.exponential(scale=1 / sumRates, size=1)[0]

        if t + dt < nextStep:

            t += dt
   

            simData = doEvent2(rates, params, simData)

        else:

            simData = doFreeLive(params, simData, nextStep - freeliveTime)

            t = nextStep
            freeliveTime = nextStep
            timeBarrier = nextStep

            # ageing and death
            if timeBarrier >= nextAgeTime:

                simData = doDeath(params, simData, t)

                nextAgeTime += ageingInt

            # chemotherapy
            if timeBarrier >= nextChemoTime1:

                simData = doDeath(params, simData, t)
                simData = doChemo(params, simData, t, params['coverage1'])

                currentchemoTiming1[nextChemoIndex1] = maxTime + 10
                nextChemoIndex1 = np.argmin(currentchemoTiming1)
                nextChemoTime1 = currentchemoTiming1[nextChemoIndex1]

            if timeBarrier >= nextChemoTime2:

                simData = doDeath(params, simData, t)
                simData = doChemo(params, simData, t, params['coverage2'])

                currentchemoTiming2[nextChemoIndex2] = maxTime + 10
                nextChemoIndex2 = np.argmin(currentchemoTiming2)
                nextChemoTime2 = currentchemoTiming2[nextChemoIndex2]

            if timeBarrier >= nextVaccineTime:

                simData = doDeath(params, simData, t)
                simData = doVaccine(params, simData, t, params['VaccCoverage'])
                currentVaccineTimings[nextVaccineIndex] = maxTime + 10
                nextVaccineIndex = np.argmin(currentVaccineTimings)
                nextVaccineTime = currentVaccineTimings[nextVaccineIndex]

            if timeBarrier >= nextOutTime:

                results.append(dict(
                    iteration=i,
                    time=t,
                    worms=copy.deepcopy(simData['worms']),
                    hosts=copy.deepcopy(simData['demography']),
                    vaccState=copy.deepcopy(simData['sv']),
                    freeLiving=copy.deepcopy(simData['freeLiving']),
                    adherenceFactors=copy.deepcopy(
                        simData['adherenceFactors']),
                    compliers=copy.deepcopy(simData['compliers'])
                ))
                outTimes[nextOutIndex] = maxTime + 10
                nextOutIndex = np.argmin(outTimes)
                nextOutTime = outTimes[nextOutIndex]

            nextStep = np.min([nextOutTime, t + maxStep, nextChemoTime1,
                              nextChemoTime2, nextVaccineTime, nextAgeTime])

    results.append(dict(  # attendanceRecord=np.array(simData['attendanceRecord']),
        # ageAtChemo=np.array(simData['ageAtChemo']),
        # adherenceFactorAtChemo=np.array(simData['adherenceFactorAtChemo'])
    ))

    return results


def SCH_Simulation(paramFileName, demogName, numReps=None):
    '''
    This function generates multiple simulation paths.

    Parameters
    ----------
    paramFileName: str
        name of the input text file with the model parameters;

    demogName: str
        subset of demography parameters to be extracted;

    numReps: int
        number of simulations;

    Returns
    -------
    df: data frame
        data frame with simulation results;
    '''
    num_cores = 1
    # initialize the parameters
    params = loadParameters(paramFileName, demogName)

    # extract the number of simulations
    if numReps is None:
        numReps = params['numReps']

    # run the simulations
    results = Parallel(n_jobs=num_cores)(
        delayed(doRealization)(params, i) for i in range(numReps))

    # process the output
    output = extractHostData(results)

    # transform the output to data frame
    df = getPrevalence(output, params, numReps)

    return df
