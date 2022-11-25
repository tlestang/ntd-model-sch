import copy
import math
import multiprocessing
import random
import time
from typing import List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sch_simulation.helsim_FUNC_KK import (
    Demography,
    Parameters,
    Result,
    SDEquilibrium,
    Worms,
    calcRates2,
    conductSurvey,
    configure,
    doChemo,
    doChemoAgeRange,
    doDeath,
    doEvent2,
    doFreeLive,
    doVaccine,
    doVaccineAgeRange,
    doVectorControl,
    extractHostData,
    getEquilibrium,
    getPrevalence,
    getPrevalenceDALYsAll,
    getPsi,
    nextMDAVaccInfo,
    outputNumberInAgeGroup,
    overWritePostMDA,
    overWritePostVacc,
    overWritePostVecControl,
    parse_coverage_input,
    readCoverageFile,
    readParams,
    setupSD,
)

num_cores = multiprocessing.cpu_count()


def loadParameters(paramFileName: str, demogName: str) -> Parameters:
    """
        This function loads all the parameters from the input text
    params    files and organizes them in a dictionary.
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
    """

    # load the parameters
    params = readParams(paramFileName=paramFileName, demogName=demogName)

    # configure the parameters
    params = configure(params)

    # update the parameters
    params.psi = getPsi(params)
    params.equiData = getEquilibrium(params)

    return params


def doRealization(params: Parameters, i: int) -> List[Result]:
    """
    This function generates a single simulation path.
    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;
    i: int
        iteration number;
    Returns
    -------
    results: List[Result]
        list with simulation results;
    """

    # setup simulation data
    simData = setupSD(params)

    # start time
    t = 0

    # end time
    maxTime = copy.deepcopy(params.maxTime)

    # time at which to update the freelive population
    freeliveTime = t

    # times at which data should be recorded
    outTimes = copy.deepcopy(params.outTimings)

    # time when data should be recorded next
    nextOutIndex = np.argmin(outTimes)
    nextOutTime = outTimes[nextOutIndex]

    # time at which individuals' age is advanced next
    ageingInt = 1 / 52
    nextAgeTime = 1 / 52
    maxStep = 1 / 52

    # time at which individuals receive next chemotherapy
    currentchemoTiming1 = copy.deepcopy(params.chemoTimings1)
    currentchemoTiming2 = copy.deepcopy(params.chemoTimings2)

    currentVaccineTimings = copy.deepcopy(params.VaccineTimings)

    nextChemoIndex1 = np.argmin(currentchemoTiming1)
    nextChemoIndex2 = np.argmin(currentchemoTiming2)
    nextVaccineIndex = np.argmin(currentVaccineTimings)

    nextChemoTime1 = currentchemoTiming1[nextChemoIndex1]
    nextChemoTime2 = currentchemoTiming2[nextChemoIndex2]
    nextVaccineTime = currentVaccineTimings[nextVaccineIndex]
    # next event
    nextStep = np.min(
        np.array(
            [
                nextOutTime,
                t + maxStep,
                nextChemoTime1,
                nextChemoTime2,
                nextAgeTime,
                nextVaccineTime,
            ]
        )
    )

    results: List[Result] = []  # initialise empty list to store results

    # run stochastic algorithm
    while t < maxTime:
        if (t * 1000 % 10) == 0:
            print(t)
        rates = calcRates2(params, simData)
        sumRates = np.sum(rates)
        cumsumRates = np.cumsum(rates)

        # if the rate is such that nothing's likely to happen in the next 10,000 years,
        # just fix the next time step to 10,000
        if sumRates < 1e-4:
            dt = 10000

        else:
            dt = random.expovariate(lambd=sumRates)

        if t + dt < nextStep:
            t += dt
            simData = doEvent2(sumRates, cumsumRates, params, simData)

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
                simData = doChemo(params, simData, t, params.coverage1)

                currentchemoTiming1[nextChemoIndex1] = maxTime + 10
                nextChemoIndex1 = np.argmin(currentchemoTiming1)
                nextChemoTime1 = currentchemoTiming1[nextChemoIndex1]

            if timeBarrier >= nextChemoTime2:

                simData = doDeath(params, simData, t)
                simData = doChemo(params, simData, t, params.coverage2)

                currentchemoTiming2[nextChemoIndex2] = maxTime + 10
                nextChemoIndex2 = np.argmin(currentchemoTiming2)
                nextChemoTime2 = currentchemoTiming2[nextChemoIndex2]

            if timeBarrier >= nextVaccineTime:

                simData = doDeath(params, simData, t)
                simData = doVaccine(params, simData, t, params.VaccCoverage)
                currentVaccineTimings[nextVaccineIndex] = maxTime + 10
                nextVaccineIndex = np.argmin(currentVaccineTimings)
                nextVaccineTime = currentVaccineTimings[nextVaccineIndex]

            if timeBarrier >= nextOutTime:

                results.append(
                    Result(
                        iteration=i,
                        time=t,
                        worms=copy.deepcopy(simData.worms),
                        hosts=copy.deepcopy(simData.demography),
                        vaccState=copy.deepcopy(simData.sv),
                        freeLiving=copy.deepcopy(simData.freeLiving),
                        adherenceFactors=copy.deepcopy(simData.adherenceFactors),
                        compliers=copy.deepcopy(simData.compliers),
                        sex_id=copy.deepcopy(simData.sex_id),
                    )
                )
                outTimes[nextOutIndex] = maxTime + 10
                nextOutIndex = np.argmin(outTimes)
                nextOutTime = outTimes[nextOutIndex]

            nextStep = np.min(
                [
                    nextOutTime,
                    t + maxStep,
                    nextChemoTime1,
                    nextChemoTime2,
                    nextVaccineTime,
                    nextAgeTime,
                ]
            )

    # results.append(dict(  # attendanceRecord=np.array(simData['attendanceRecord']),
    #    # ageAtChemo=np.array(simData['ageAtChemo']),
    #    # adherenceFactorAtChemo=np.array(simData['adherenceFactorAtChemo'])
    # ))

    return results


def doRealizationSurvey(params: Parameters, i: int) -> List[Result]:
    """
    This function generates a single simulation path.
    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;
    i: int
        iteration number;
    Returns
    -------
    results: list
        list with simulation results;
    """

    # setup simulation data
    simData = setupSD(params)

    # start time
    t = 0

    # end time
    maxTime = copy.deepcopy(params.maxTime)

    # time at which to update the freelive population
    freeliveTime = t

    # times at which data should be recorded
    outTimes = copy.deepcopy(params.outTimings)

    # time when data should be recorded next
    nextOutIndex = np.argmin(outTimes)
    nextOutTime = outTimes[nextOutIndex]

    # time at which individuals' age is advanced next
    ageingInt = 1 / 52
    nextAgeTime = 1 / 52
    maxStep = 1 / 52

    # time at which individuals receive next chemotherapy
    currentchemoTiming1 = copy.deepcopy(params.chemoTimings1)
    currentchemoTiming2 = copy.deepcopy(params.chemoTimings2)

    currentVaccineTimings = copy.deepcopy(params.VaccineTimings)

    nextChemoIndex1 = np.argmin(currentchemoTiming1)
    nextChemoIndex2 = np.argmin(currentchemoTiming2)
    nextVaccineIndex = np.argmin(currentVaccineTimings)

    nextChemoTime1 = currentchemoTiming1[nextChemoIndex1]
    nextChemoTime2 = currentchemoTiming2[nextChemoIndex2]
    nextVaccineTime = currentVaccineTimings[nextVaccineIndex]
    # next event
    nextStep = np.min(
        [
            nextOutTime,
            t + maxStep,
            nextChemoTime1,
            nextChemoTime2,
            nextAgeTime,
            nextVaccineTime,
        ]
    )

    # passSurveyTwo = 0
    nChemo = 0
    nVacc = 0
    nSurvey = 0
    # nSurveyTwo = 0
    tSurvey = maxTime + 10
    # tSurveyTwo = maxTime + 10
    results = []  # initialise empty list to store results

    # run stochastic algorithm
    while t < maxTime:

        rates = calcRates2(params, simData)
        sumRates = np.sum(rates)
        cumsumRates = np.cumsum(rates)
        # if the rate is such that nothing's likely to happen in the next 10,000 years,
        # just fix the next time step to 10,000
        if sumRates < 1e-4:

            dt = 10000

        else:

            dt = np.random.exponential(scale=1 / sumRates, size=1)[0]

        if t + dt < nextStep:

            t += dt

            simData = doEvent2(sumRates, cumsumRates, params, simData)

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
                simData = doChemo(params, simData, t, params.coverage1)
                if nChemo == 0:
                    tSurvey = t + 5
                #  tSurveyTwo = t + 9
                nChemo += 1
                currentchemoTiming1[nextChemoIndex1] = maxTime + 10
                nextChemoIndex1 = np.argmin(currentchemoTiming1)
                nextChemoTime1 = currentchemoTiming1[nextChemoIndex1]

            if timeBarrier >= nextChemoTime2:

                simData = doDeath(params, simData, t)
                simData = doChemo(params, simData, t, params.coverage2)
                if nChemo == 0:
                    tSurvey = t + 5
                nChemo += 1
                currentchemoTiming2[nextChemoIndex2] = maxTime + 10
                nextChemoIndex2 = np.argmin(currentchemoTiming2)
                nextChemoTime2 = currentchemoTiming2[nextChemoIndex2]

            if timeBarrier >= nextVaccineTime:

                simData = doDeath(params, simData, t)
                simData = doVaccine(params, simData, t, params.VaccCoverage)
                nVacc += 1
                currentVaccineTimings[nextVaccineIndex] = maxTime + 10
                nextVaccineIndex = np.argmin(currentVaccineTimings)
                nextVaccineTime = currentVaccineTimings[nextVaccineIndex]

            if timeBarrier >= tSurvey:
                simData, prevOne = conductSurvey(
                    simData, params, t, params.sampleSizeOne, params.nSamples
                )
                nSurvey += 1
                # tSurvey = maxTime + 10
                if prevOne < 0.01:
                    nextChemoTime1 = maxTime + 10
                    nextChemoTime2 = maxTime + 10
                    nextVaccineTime = maxTime + 10
                    tSurvey = maxTime + 10
                else:
                    tSurvey = t + 4
                # simData, prevTwo = conductSurveyTwo(simData, params, t, sampleSizeTwo, nSamples)
                # nSurveyTwo += 1
                # if prevTwo < 0.01:
                #     nextChemoTime1 = maxTime + 10
                #     nextChemoTime2 = maxTime + 10
                #     nextVaccineTime = maxTime + 10
                #   tSurveyTwo = maxTime + 10

            # if timeBarrier >= tSurveyTwo:
            #     simData, prevTwo = conductSurveyTwo(simData, params, t, sampleSizeTwo, nSamples)
            #     nSurveyTwo += 1
            #     if prevTwo < 0.01:
            #         nextChemoTime1 = maxTime + 10
            #         nextChemoTime2 = maxTime + 10
            #         nextVaccineTime = maxTime + 10
            #         tSurveyTwo = maxTime + 10
            #     else:
            #         tSurveyTwo = t + 4

            if timeBarrier >= nextOutTime:

                results.append(
                    Result(
                        iteration=i,
                        time=t,
                        worms=copy.deepcopy(simData.worms),
                        hosts=copy.deepcopy(simData.demography),
                        vaccState=copy.deepcopy(simData.sv),
                        freeLiving=copy.deepcopy(simData.freeLiving),
                        adherenceFactors=copy.deepcopy(simData.adherenceFactors),
                        compliers=copy.deepcopy(simData.compliers),
                        nVacc=nVacc,
                        nChemo=nChemo,
                        nSurvey=nSurvey  # ,
                        #  nSurveyTwo = nSurveyTwo
                    )
                )
                outTimes[nextOutIndex] = maxTime + 10
                nextOutIndex = np.argmin(outTimes)
                nextOutTime = outTimes[nextOutIndex]

            nextStep = np.min(
                [
                    nextOutTime,
                    t + maxStep,
                    nextChemoTime1,
                    nextChemoTime2,
                    nextVaccineTime,
                    nextAgeTime,
                ]
            )

    # results.append(dict(  # attendanceRecord=np.array(simData['attendanceRecord']),
    #     # ageAtChemo=np.array(simData['ageAtChemo']),
    #     # adherenceFactorAtChemo=np.array(simData['adherenceFactorAtChemo'])
    # ))

    return results


def doRealizationSurveyCoverage(params: Parameters, i: int) -> List[Result]:
    """
    This function generates a single simulation path.
    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;
    i: int
        iteration number;
    Returns
    -------
    results: list
        list with simulation results;
    """

    # setup simulation data
    simData = setupSD(params)

    # start time
    t = 0

    # end time
    maxTime = copy.deepcopy(params.maxTime)

    # time at which to update the freelive population
    freeliveTime = t

    # times at which data should be recorded
    outTimes = copy.deepcopy(params.outTimings)

    # time when data should be recorded next
    nextOutIndex = np.argmin(outTimes)
    nextOutTime = outTimes[nextOutIndex]

    # time at which individuals' age is advanced next
    ageingInt = 1 / 52
    nextAgeTime = 1 / 52
    maxStep = 1 / 52

    (
        chemoTiming,
        VaccTiming,
        nextChemoTime,
        nextMDAAge,
        nextChemoIndex,
        nextVaccTime,
        nextVaccAge,
        nextVaccIndex,
    ) = nextMDAVaccInfo(params)

    # next event

    nextStep = np.min(
        [nextOutTime, t + maxStep, nextChemoTime, nextAgeTime, nextVaccTime]
    )

    nChemo = 0
    nVacc = 0
    nSurvey = 0

    tSurvey = maxTime + 10
    results: List[Result] = []  # initialise empty list to store results
    print_t_interval = 0.5
    print_t = 0
    # run stochastic algorithm
    while t < maxTime:
        if t > print_t:
            print(t)
            print_t += print_t_interval
        rates = calcRates2(params, simData)
        sumRates = np.sum(rates)
        cumsumRates = np.cumsum(rates)
        # if the rate is such that nothing's likely to happen in the next 10,000 years,
        # just fix the next time step to 10,000
        if sumRates < 1e-4:

            dt = 10000

        else:
            dt = np.random.exponential(scale=1 / sumRates, size=1)[0]

        if t + dt < nextStep:
            t += dt
            simData = doEvent2(sumRates, cumsumRates, params, simData)

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
            if timeBarrier >= nextChemoTime:

                simData = doDeath(params, simData, t)
                assert params.MDA is not None
                for i in range(len(nextMDAAge)):
                    k = nextMDAAge[i] - 1
                    index = nextChemoIndex[i]
                    cov = params.MDA[k].Coverage[index]
                    minAge = params.MDA[k].Age[0]
                    maxAge = params.MDA[k].Age[1]
                    simData = doChemoAgeRange(params, simData, t, minAge, maxAge, cov)
                if nChemo == 0:
                    tSurvey = t + params.timeToFirstSurvey
                nChemo += 1

                params = overWritePostMDA(params, nextMDAAge, nextChemoIndex)

                (
                    chemoTiming,
                    VaccTiming,
                    nextChemoTime,
                    nextMDAAge,
                    nextChemoIndex,
                    nextVaccTime,
                    nextVaccAge,
                    nextVaccIndex,
                ) = nextMDAVaccInfo(params)

            # vaccination
            if timeBarrier >= nextVaccTime:

                simData = doDeath(params, simData, t)
                assert params.Vacc is not None
                for i in range(len(nextVaccAge)):
                    k = nextVaccAge[i] - 1
                    index = nextVaccIndex[i]
                    cov = params.Vacc[k].Coverage[index]
                    minAge = params.Vacc[k].Age[0]
                    maxAge = params.Vacc[k].Age[1]
                    simData = doVaccineAgeRange(params, simData, t, minAge, maxAge, cov)

                nVacc += 1
                params = overWritePostVacc(params, nextVaccAge, nextVaccIndex)

                (
                    chemoTiming,
                    VaccTiming,
                    nextChemoTime,
                    nextMDAAge,
                    nextChemoIndex,
                    nextVaccTime,
                    nextVaccAge,
                    nextVaccIndex,
                ) = nextMDAVaccInfo(params)

            # survey
            if timeBarrier >= tSurvey:
                simData, prevOne = conductSurvey(
                    simData, params, t, params.sampleSizeOne, params.nSamples
                )
                nSurvey += 1
                assert params.MDA is not None
                assert params.Vacc is not None
                if prevOne < params.surveyThreshold:
                    for mda in params.MDA:
                        mda.Years = np.array([maxTime + 10])
                    for vacc in params.Vacc:
                        vacc.Years = np.array([maxTime + 10])

                    tSurvey = maxTime + 10
                else:
                    tSurvey = t + params.timeToNextSurvey

                (
                    chemoTiming,
                    VaccTiming,
                    nextChemoTime,
                    nextMDAAge,
                    nextChemoIndex,
                    nextVaccTime,
                    nextVaccAge,
                    nextVaccIndex,
                ) = nextMDAVaccInfo(params)

            if timeBarrier >= nextOutTime:

                results.append(
                    Result(
                        iteration=i,
                        time=t,
                        worms=copy.deepcopy(simData.worms),
                        hosts=copy.deepcopy(simData.demography),
                        vaccState=copy.deepcopy(simData.sv),
                        freeLiving=copy.deepcopy(simData.freeLiving),
                        adherenceFactors=copy.deepcopy(simData.adherenceFactors),
                        compliers=copy.deepcopy(simData.compliers),
                        nVacc=nVacc,
                        nChemo=nChemo,
                        nSurvey=nSurvey,
                    )
                )
                outTimes[nextOutIndex] = maxTime + 10
                nextOutIndex = np.argmin(outTimes)
                nextOutTime = outTimes[nextOutIndex]

            nextStep = np.min(
                [nextOutTime, t + maxStep, nextChemoTime, nextAgeTime, nextVaccTime]
            )

    # results.append(dict(  # attendanceRecord=np.array(simData['attendanceRecord']),
    #    # ageAtChemo=np.array(simData['ageAtChemo']),
    #    # adherenceFactorAtChemo=np.array(simData['adherenceFactorAtChemo'])
    # ))

    return results


def doRealizationSurveyCoveragePickle(
    params: Parameters, simData: SDEquilibrium, i: int
) -> List[Result]:
    """
    This function generates a single simulation path.

    Parameters
    ----------
    params: Parameters
        dataclass containing the parameter names and values;

    simData: SDEquilibrium
        dataclass containing the initial equilibrium parameter values;

    i: int
        iteration number;

    Returns
    -------
    results: list
        list with simulation results;
    """
    
    # start time
    t: float = 0

    # end time
    maxTime = params.maxTime

    # time at which to update the freelive population
    freeliveTime = t

    # times at which data should be recorded
    outTimes = copy.deepcopy(params.outTimings)

    # time when data should be recorded next
    nextOutIndex = np.argmin(outTimes)
    nextOutTime = outTimes[nextOutIndex]

    # time at which individuals' age is advanced next
    ageingInt = 1 / 52
    nextAgeTime = 1 / 52
    maxStep = 1 / 52

    (
        chemoTiming,
        VaccTiming,
        nextChemoTime,
        nextMDAAge,
        nextChemoIndex,
        nextVaccTime,
        nextVaccAge,
        nextVaccIndex,
        nextVecControlTime,
        nextVecControlIndex,
    ) = nextMDAVaccInfo(params)

    # next event
    
    nextStep = min(
        float(nextOutTime),
        float(t + maxStep),
        float(nextChemoTime),
        float(nextAgeTime),
        float(nextVaccTime),
        float(nextVecControlTime)
    )
    
    propChemo1 = []
    propChemo2 = []
    propVacc = []
    prevNChemo1 = 0
    prevNChemo2 = 0
    prevNVacc = 0
    nChemo = 0
    nVacc = 0
    nSurvey = 0
    surveyPass = 0
    tSurvey = maxTime + 10
    results = []  # initialise empty list to store results
    print_t_interval = 0.5
    print_t = 0
  
    # run stochastic algorithm
    multiplier = math.floor(
        params.N / 50
    )  # This appears to be the optimal value for all tests I've run - more or less than this takes longer!
    while t < maxTime:
        
        if t > print_t:
            print_t += print_t_interval
            #print(t)
        rates = calcRates2(params, simData)
        sumRates = np.sum(rates)
        cumsumRates = np.cumsum(rates)
        # If the nextStep is soon, take a smaller multiplier
        # new_multiplier = max(math.floor(min((nextStep - t) * sumRates, multiplier)), 1)
        new_multiplier = min(math.floor(min((1/365) * sumRates, multiplier)), 10)
        new_multiplier = max(new_multiplier, 10)
        #new_multiplier = 1
        # if the rate is such that nothing's likely to happen in the next 10,000 years,
        # just fix the next time step to 10,000

        if sumRates < 1e-4:
            dt = 10000 * new_multiplier
        else:
            dt = random.expovariate(lambd=sumRates) * new_multiplier

        new_t = t + dt
        if new_t < nextStep:
            t = new_t
            simData = doEvent2(sumRates, cumsumRates, params, simData, new_multiplier)
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
            if timeBarrier >= nextChemoTime:

                simData = doDeath(params, simData, t)
                assert params.MDA is not None
                for i in range(len(nextMDAAge)):
                    k = nextMDAAge[i]
                    index = nextChemoIndex[i]
                    cov = params.MDA[k].Coverage[index]
                    minAge = params.MDA[k].Age[0]
                    maxAge = params.MDA[k].Age[1]
                    simData, propTreated1, propTreated2 = doChemoAgeRange(params, simData, t, minAge, maxAge, cov)
                    if propTreated1 > 0:
                        propChemo1.append([t, minAge, maxAge, "C1", round(propTreated1,4)])
                    if propTreated2 > 0:
                        propChemo2.append([t, minAge, maxAge, "C2", round(propTreated2,4)])
                    
                if nChemo == 0:
                    tSurvey = t + params.timeToFirstSurvey
                nChemo += 1

                params = overWritePostMDA(params, nextMDAAge, nextChemoIndex)

                (
                    chemoTiming,
                    VaccTiming,
                    nextChemoTime,
                    nextMDAAge,
                    nextChemoIndex,
                    nextVaccTime,
                    nextVaccAge,
                    nextVaccIndex,
                    nextVecControlTime,
                    nextVecControlIndex,
                ) = nextMDAVaccInfo(params)

            # vaccination
            if timeBarrier >= nextVaccTime:
           #     break
                simData = doDeath(params, simData, t)
                assert params.Vacc is not None
                for i in range(len(nextVaccAge)):
                    k = nextVaccAge[i]
                    index = nextVaccIndex[i]
                    cov = params.Vacc[k].Coverage[index]
                    minAge = params.Vacc[k].Age[0]
                    maxAge = params.Vacc[k].Age[1]
                    simData, pVacc = doVaccineAgeRange(params, simData, t, minAge, maxAge, cov)
                    propVacc.append([t, minAge, maxAge, round(pVacc,4)])
                    
                nVacc += 1
                params = overWritePostVacc(params, nextVaccAge, nextVaccIndex)

                (
                    chemoTiming,
                    VaccTiming,
                    nextChemoTime,
                    nextMDAAge,
                    nextChemoIndex,
                    nextVaccTime,
                    nextVaccAge,
                    nextVaccIndex,
                    nextVecControlTime,
                    nextVecControlIndex,
                ) = nextMDAVaccInfo(params)
            
            
            if timeBarrier >= nextVecControlTime:
                cov = params.VecControl[0].Coverage[nextVecControlIndex]
                simData = doVectorControl(params, simData, cov)
                params = overWritePostVecControl(params, nextVecControlIndex)
                
                (
                    chemoTiming,
                    VaccTiming,
                    nextChemoTime,
                    nextMDAAge,
                    nextChemoIndex,
                    nextVaccTime,
                    nextVaccAge,
                    nextVaccIndex,
                    nextVecControlTime,
                    nextVecControlIndex,
                ) = nextMDAVaccInfo(params)
            # survey
            if timeBarrier >= tSurvey:
                simData, prevOne = conductSurvey(
                    simData, params, t, params.sampleSizeOne, params.nSamples
                )
                nSurvey += 1

                if prevOne < params.surveyThreshold:
                    surveyPass = 1
                    assert params.MDA is not None
                    for mda in params.MDA:
                        mda.Years = np.array([maxTime + 10])
                    assert params.Vacc is not None
                    for vacc in params.Vacc:
                        vacc.Years = np.array([maxTime + 10])
                        
                    tSurvey = maxTime + 10
                else:
                    tSurvey = t + params.timeToNextSurvey

                (
                    chemoTiming,
                    VaccTiming,
                    nextChemoTime,
                    nextMDAAge,
                    nextChemoIndex,
                    nextVaccTime,
                    nextVaccAge,
                    nextVaccIndex,
                    nextVecControlTime,
                    nextVecControlIndex,
                ) = nextMDAVaccInfo(params)

            if timeBarrier >= nextOutTime:


                a, truePrev = conductSurvey(simData, params, t, params.N, 2)
                trueElim = int(1 - truePrev)
                
                results.append(
                    Result(
                        iteration=i,
                        time=t,
                        worms=copy.deepcopy(simData.worms),
                        hosts=copy.deepcopy(simData.demography),
                        vaccState=copy.deepcopy(simData.sv),
                        freeLiving=copy.deepcopy(simData.freeLiving),
                        adherenceFactors=copy.deepcopy(simData.adherenceFactors),
                        compliers=copy.deepcopy(simData.compliers),
                        si=copy.deepcopy(simData.si),
                        sv=copy.deepcopy(simData.sv),
                        contactAgeGroupIndices=copy.deepcopy(simData.contactAgeGroupIndices),
                        nVacc=simData.vaccCount - prevNVacc,
                        nChemo1=simData.nChemo1 - prevNChemo1,
                        nChemo2=simData.nChemo2 - prevNChemo2,
                        nSurvey=nSurvey,
                        surveyPass=surveyPass,
                        elimination=trueElim,
                        propChemo1=propChemo1,
                        propChemo2=propChemo2,
                        propVacc = propVacc
                    )
                )
                prevNChemo1 = simData.nChemo1 
                prevNChemo2 = simData.nChemo2
                prevNVacc = simData.vaccCount
                outTimes[nextOutIndex] = maxTime + 10
                nextOutIndex = np.argmin(outTimes)
                nextOutTime = outTimes[nextOutIndex]

            nextStep = min(
                float(nextOutTime),
                float(t + maxStep),
                float(nextChemoTime),
                float(nextAgeTime),
                float(nextVaccTime),
                float(nextVecControlTime),
            )
  
    # results.append(dict(  # attendanceRecord=np.array(simData['attendanceRecord']),
    #     # ageAtChemo=np.array(simData['ageAtChemo']),
    #     # adherenceFactorAtChemo=np.array(simData['adherenceFactorAtChemo'])
    # ))
  
    return results, simData


def SCH_Simulation(
    paramFileName: str, demogName: str, numReps: Optional[int] = None
) -> pd.DataFrame:
    """
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
    """

    # initialize the parameters
    params = loadParameters(paramFileName, demogName)

    # extract the number of simulations
    if numReps is None:
        numReps = params.numReps

    # run the simulations
    results = Parallel(n_jobs=num_cores)(
        delayed(doRealization)(params, i) for i in range(numReps)
    )

    # process the output
    output = extractHostData(results)

    # transform the output to data frame
    df = getPrevalence(output, params, numReps)

    return df


def SCH_Simulation_DALY(
    paramFileName: str, demogName: str, numReps: Optional[int] = None
) -> pd.DataFrame:
    """
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
    """

    # initialize the parameters
    params = loadParameters(paramFileName, demogName)

    # extract the number of simulations
    if numReps is None:
        numReps = params.numReps

    # run the simulations
    results = Parallel(n_jobs=num_cores)(
        delayed(doRealization)(params, i) for i in range(numReps)
    )

    # process the output
    output = extractHostData(results)

    # transform the output to data frame
    df = getPrevalenceDALYsAll(output, params, numReps)

    return df



def getActualCoverages(results: List[List[Result]], params: Parameters, allTimes)-> pd.DataFrame:
    
    ind = 0
    p = copy.deepcopy(params.MDA)
    for i in range(len(p)):
        a1 = p[i].Age[0]
        a2 = p[i].Age[1]
        if i == 0:
            df1 = pd.DataFrame(
                        {
                            "Time": allTimes,
                            "age_start": np.repeat(a1, len(allTimes)),
                            "age_end": np.repeat(a2, len(allTimes)),
                            "intensity": np.repeat("None", len(allTimes)),
                            "species": np.repeat(params.species, len(allTimes)),
                            "measure": np.repeat("Chemo1Cov", len(allTimes)),
                            "draw_1": np.repeat(0, len(allTimes)),
                        }
                    )
                                                 
        else:
            df1 = df1.append(pd.DataFrame(
                        {
                            "Time": allTimes,
                            "age_start": np.repeat(a1, len(allTimes)),
                            "age_end": np.repeat(a2, len(allTimes)),
                            "intensity": np.repeat("None", len(allTimes)),
                            "species": np.repeat(params.species, len(allTimes)),
                            "measure": np.repeat("Chemo1Cov", len(allTimes)),
                            "draw_1": np.repeat(0, len(allTimes)),
                        }
                    ))
            
        df1 = df1.append(pd.DataFrame(
                        {
                            "Time": allTimes,
                            "age_start": np.repeat(a1, len(allTimes)),
                            "age_end": np.repeat(a2, len(allTimes)),
                            "intensity": np.repeat("None", len(allTimes)),
                            "species": np.repeat(params.species, len(allTimes)),
                            "measure": np.repeat("Chemo2Cov", len(allTimes)),
                            "draw_1": np.repeat(0, len(allTimes)),
                        }
                    ))
    p = copy.deepcopy(params.Vacc)
    for i in range(len(p)):
        a1 = p[i].Age[0]
        a2 = p[i].Age[1]
        
            
        df1 = df1.append(pd.DataFrame(
                        {
                            "Time": allTimes,
                            "age_start": np.repeat(a1, len(allTimes)),
                            "age_end": np.repeat(a2, len(allTimes)),
                            "intensity": np.repeat("None", len(allTimes)),
                            "species": np.repeat(params.species, len(allTimes)),
                            "measure": np.repeat("VaccCov", len(allTimes)),
                            "draw_1": np.repeat(0, len(allTimes)),
                        }
                    ))

        
    pp = len(results[0])
    for i in range(len(results[0][pp-1].propChemo1)):
        df = results[0][pp-1].propChemo1[i]
        t = df[0]
        a1 = df[1]
        a2 = df[2]
        k = np.where(np.logical_and(df1.measure == "Chemo1Cov", np.logical_and(np.logical_and(df1.Time == t, df1.age_start == a1), df1.age_end == a2)))
        
        df1.draw_1.iloc[k] = df[4]
        
            
    for i in range(len(results[0][pp-1].propChemo2)):
        df = results[0][pp-1].propChemo2[i]
        t = df[0]
        a1 = df[1]
        a2 = df[2]
        k = np.where(np.logical_and(df1.measure == "Chemo2Cov", np.logical_and(np.logical_and(df1.Time == t, df1.age_start == a1), df1.age_end == a2)))
        
        df1.draw_1.iloc[k] = df[4]
        
            
    for i in range(len(results[0][pp-1].propVacc)):
        df = results[0][pp-1].propVacc[i]
        t = df[0]
        a1 = df[1]
        a2 = df[2]
        k = np.where(np.logical_and(df1.measure == "VaccCov", np.logical_and(np.logical_and(df1.Time == t, df1.age_start == a1), df1.age_end == a2)))
        
        df1.draw_1.iloc[k] = df[3]
        
            
    return df1
        
def getCostData(results: List[List[Result]], params: Parameters) -> pd.DataFrame:
    df1 = None
    for i, list_res in enumerate(results):
        df = pd.DataFrame(list_res)
        if i == 0:
            df1 = pd.DataFrame(
                {
                    "Time": df["time"],
                    "age_start": np.repeat("None", df.shape[0]),
                    "age_end": np.repeat("None", df.shape[0]),
                    "intensity": np.repeat("None", df.shape[0]),
                    "species": np.repeat(params.species, df.shape[0]),
                    "measure": np.repeat("nChemo1", df.shape[0]),
                    "draw_1": df["nChemo1"],
                }
            )
        else:
            assert df1 is not None
            df1 = df1.append(
                pd.DataFrame(
                    {
                        "Time": df["time"],
                        "age_start": np.repeat("None", df.shape[0]),
                        "age_end": np.repeat("None", df.shape[0]),
                        "intensity": np.repeat("None", df.shape[0]),
                        "species": np.repeat(params.species, df.shape[0]),
                        "measure": np.repeat("nChemo", df.shape[0]),
                        "draw_1": df["nChemo"],
                    }
                )
            )
        df1 = df1.append(
            pd.DataFrame(
                {
                    "Time": df["time"],
                    "age_start": np.repeat("None", df.shape[0]),
                    "age_end": np.repeat("None", df.shape[0]),
                    "intensity": np.repeat("None", df.shape[0]),
                    "species": np.repeat(params.species, df.shape[0]),
                    "measure": np.repeat("nChemo2", df.shape[0]),
                    "draw_1": df["nChemo2"],
                }
            )
        )
        df1 = df1.append(
            pd.DataFrame(
                {
                    "Time": df["time"],
                    "age_start": np.repeat("None", df.shape[0]),
                    "age_end": np.repeat("None", df.shape[0]),
                    "intensity": np.repeat("None", df.shape[0]),
                    "species": np.repeat(params.species, df.shape[0]),
                    "measure": np.repeat("nVacc", df.shape[0]),
                    "draw_1": df["nVacc"],
                }
            )
        )
        df1 = df1.append(
            pd.DataFrame(
                {
                    "Time": df["time"],
                    "age_start": np.repeat("None", df.shape[0]),
                    "age_end": np.repeat("None", df.shape[0]),
                    "intensity": np.repeat("None", df.shape[0]),
                    "species": np.repeat(params.species, df.shape[0]),
                    "measure": np.repeat("nSurvey", df.shape[0]),
                    "draw_1": df["nSurvey"],
                }
            )
        )
        df1 = df1.append(
            pd.DataFrame(
                {
                    "Time": df["time"],
                    "age_start": np.repeat("None", df.shape[0]),
                    "age_end": np.repeat("None", df.shape[0]),
                    "intensity": np.repeat("None", df.shape[0]),
                    "species": np.repeat(params.species, df.shape[0]),
                    "measure": np.repeat("surveyPass", df.shape[0]),
                    "draw_1": df["surveyPass"],
                }
            )
        )
        df1 = df1.append(
            pd.DataFrame(
                {
                    "Time": df["time"],
                    "age_start": np.repeat("None", df.shape[0]),
                    "age_end": np.repeat("None", df.shape[0]),
                    "intensity": np.repeat("None", df.shape[0]),
                    "species": np.repeat(params.species, df.shape[0]),
                    "measure": np.repeat("trueElimination", df.shape[0]),
                    "draw_1": df["elimination"],
                }
            )
        )
    return df1


def SCH_Simulation_DALY_Coverage(
    paramFileName: str,
    demogName: str,
    coverageFileName: str,
    coverageTextFileStorageName: str,
    numReps: Optional[int] = None,
) -> pd.DataFrame:
    """
    This function generates multiple simulation paths.
    Parameters
    ----------
    paramFileName: str
        name of the input text file with the model parameters;
    demogName: str
        subset of demography parameters to be extracted;
    numReps: Optional[int]
        number of simulations - if none is params.numReps;
    Returns
    -------
    df: data frame
        data frame with simulation results;
    """
    parse_coverage_input(coverageFileName, coverageTextFileStorageName)
    # initialize the parameters
    params = loadParameters(paramFileName, demogName)
    params = readCoverageFile(coverageTextFileStorageName, params)
    # extract the number of simulations
    if numReps is None:
        numReps = params.numReps

    # run the simulations
    results = Parallel(n_jobs=num_cores)(
        delayed(doRealizationSurveyCoverage)(params, i) for i in range(numReps)
    )

    # process the output
    output = extractHostData(results)

    # transform the output to data frame
    df = getPrevalenceDALYsAll(output, params, numReps)

    return df


def singleSimulationDALYCoverage(
    params: Parameters, simData: SDEquilibrium, numReps: Optional[int] = None
) -> pd.DataFrame:
    """
    This function generates multiple simulation paths.
    Parameters
    ----------
    params : Parameters
        set of parameters for the run
    Returns
    -------
    df: data frame
        data frame with simulation results;
    """

    # extract the number of simulations
    if numReps is None:
        numReps = params.numReps

    # run the simulations
    results, SD = doRealizationSurveyCoveragePickle(params, simData, 1)
    
 
    results = [results]
    # process the output
    output = extractHostData(results)

    # transform the output to data frame
    df = getPrevalenceDALYsAll(
        output, params, numReps, Unfertilized=params.Unfertilized
    )
         
     
    numAgeGroup = outputNumberInAgeGroup(results, params)
    costData = getCostData(results, params)
    allTimes = np.unique(numAgeGroup.Time)
    trueCoverageData = getActualCoverages(results, params, allTimes)
    df1 = pd.concat([df, numAgeGroup], ignore_index=True)
    df1 = pd.concat([df1, costData], ignore_index=True)
    df1 = pd.concat([df1, trueCoverageData], ignore_index=True)
    df1 = df1.reset_index()
    df1['draw_1'][np.where(pd.isna(df1['draw_1']))[0]] = -1
    df1 = df1[['Time','age_start','age_end', 'intensity', 'species', 'measure', 'draw_1']]
    return df1, SD



def getDesiredAgeDistribution(params, timeLimit):
    t = 0
    # initialize birth and death ages from population
    deathAges = getLifeSpans(params.N, params)
    birthAges = np.zeros(params.N)
    # age population for a chosen number of years in order to generate 
    # wanted age distribution
    while t < timeLimit:
        t = min(deathAges)
        theDead = np.where(deathAges == t)[0]
        newDeathAges = getLifeSpans(len(theDead), params)
        deathAges[theDead] = newDeathAges + t
        birthAges[theDead] =  t
    ages = t - birthAges
    deathAges = deathAges - t
    aa = np.zeros([len(ages),2])
    aa[:,0] = ages
    aa[:,1] = deathAges
    b = aa
    ord1 = aa[:, 0].argsort()
    b[:,0] = aa[ord1,0]
    b[:,1] = deathAges[ord1]
    return b

def splitSimDataIntoAges(ages, ageGroups):
    # this function should return an array which contains the location of individuals in the pickle
    # file who have age between 2 ages
    # ages is ages in pickle file
    # ageGroups is different splits in the ages to choose individuals from
    groupAges = []
    for i in range(1, len(ageGroups)):
        k1 = ages >= ageGroups[i-1]
        k2 = ages < ageGroups[i]
        k = np.where(np.logical_and(k1, k2))
        
        groupAges.append(dict(
                              minAge = ageGroups[i-1],maxAge= ageGroups[i],
                         indices = k[0]))
    
    return groupAges


def findNumberOfPeopleEachAgeGroup(chosenAges, groupAges):
    # how many people in each age group are in the wanted distribution
    # chosenAges is the age of people in the wanted distribution
    numIndivsToChoose = []
    for i in range(len(groupAges)):
        minAge = groupAges[i]['minAge']
        maxAge = groupAges[i]['maxAge']
        k1 = chosenAges >= minAge
        k2 = chosenAges < maxAge
        numIndivsToChoose.append(len(np.where(np.logical_and(k1,k2))[0]))
    return numIndivsToChoose


def selectIndividuals(chosenAges,  groupAges, numIndivsToChoose):
    chosenIndivs = np.zeros(sum(numIndivsToChoose) ,dtype = int)
    startPoint = 0
    totalPeople = 0
    for i in range(len(numIndivsToChoose)):
        n1 = numIndivsToChoose[i]
        indivs = np.array(groupAges[i]['indices'])
        chosenIndivs[range(startPoint,startPoint+n1)] =  np.random.choice(a=indivs, size=n1, replace=True)
        startPoint += n1
    return chosenIndivs



def multiple_simulations(
    params: Parameters, pickleData, simparams, indices, i, wantedPopSize = 3000,
    ageGroups = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 100]
) -> pd.DataFrame:
    print(f"==> multiple_simulations starting sim {i}")
    start_time = time.time()
    # copy the parameters
    parameters = copy.deepcopy(params)
    j = indices[i]
    # load the previous simulation results
    data = pickleData[j]

    # extract the previous simulation output
    keys = [
        "si",
        "worms",
        "freeLiving",
        "demography",
        "contactAgeGroupIndices",
        "treatmentAgeGroupIndices",
    ]
    t = 0

    raw_data = dict((key, copy.deepcopy(data[key])) for key in keys)
    times = data["times"]
    raw_data["demography"]["birthDate"] = raw_data["demography"]["birthDate"] - times["maxTime"]
    raw_data["demography"]["deathDate"] = raw_data["demography"]["deathDate"]- times["maxTime"]
    worms = Worms(total=raw_data["worms"]["total"], female=raw_data["worms"]["female"])
    demography = Demography(
        birthDate=raw_data["demography"]["birthDate"],
        deathDate=raw_data["demography"]["deathDate"],
    )
    simData = SDEquilibrium(
        si=raw_data["si"],
        worms=worms,
        freeLiving=raw_data["freeLiving"],
        demography=demography,
        contactAgeGroupIndices=raw_data["contactAgeGroupIndices"],
        treatmentAgeGroupIndices=raw_data["treatmentAgeGroupIndices"],
        sv=np.zeros(len(raw_data["si"]), dtype=int),
        attendanceRecord=[],
        ageAtChemo=[],
        adherenceFactorAtChemo=[],
        vaccCount=0,
        nChemo1=0,
        nChemo2=0,
        numSurvey=0,
        compliers=np.random.uniform(low=0, high=1, size=len(raw_data["si"]))
        > params.propNeverCompliers,
        adherenceFactors=np.random.uniform(low=0, high=1, size=len(raw_data["si"])),
    )
    pickleNumIndivs = len(simData.si)
    #print("starting  j =",j)
    if pickleNumIndivs < wantedPopSize:
        # set population size to wanted population size
        params.N = wantedPopSize
        # get the birth and death ages of representative population
        b = getDesiredAgeDistribution(params, timeLimit = 200)
        chosenAges = b[:, 0]
        deathDate = b[:, 1]
        # ages of pickle file data
        ages = -simData.demography.birthDate
        # group these ages into age groups
        groupAges = splitSimDataIntoAges(ages, ageGroups)
        # how many people in each age group do we need to pick to match representative population
        numIndivsToChoose = findNumberOfPeopleEachAgeGroup(chosenAges,  groupAges)
        # choose these people from the pickle data
        chosenIndivs = selectIndividuals(chosenAges, groupAges, numIndivsToChoose)
        birthDate = -chosenAges - 0.000001
        deathDate = deathDate 
        wormsT = []
        wormsF = []
        si = []
        contactAgeGroupIndices = []
        treatmentAgeGroupIndices = []
        for k in range(len(chosenIndivs)):
            l = chosenIndivs[k]
            si.append(simData.si[l])
            wormsT.append(simData.worms.total[l])
            wormsF.append(simData.worms.female[l])
            contactAgeGroupIndices.append(simData.contactAgeGroupIndices[l])
            treatmentAgeGroupIndices.append(simData.treatmentAgeGroupIndices[l])
        demography = Demography(
        birthDate=np.array(birthDate),
        deathDate=np.array(deathDate),
        )
        worms = Worms(total=np.array(wormsT), female=np.array(wormsF))

        SD = SDEquilibrium(
        si=np.array(si),
        worms=worms,
        freeLiving=raw_data["freeLiving"],
        demography=demography,
        contactAgeGroupIndices=np.array(contactAgeGroupIndices),
        treatmentAgeGroupIndices=np.array(treatmentAgeGroupIndices),
        sv=np.zeros(wantedPopSize, dtype=int),
        attendanceRecord=[],
        ageAtChemo=[],
        adherenceFactorAtChemo=[],
        vaccCount=0,
        nChemo1=0,
        nChemo2=0,
        numSurvey=0,
        compliers=np.random.uniform(low=0, high=1, size=wantedPopSize)
        > params.propNeverCompliers,
        adherenceFactors=np.random.uniform(low=0, high=1, size=wantedPopSize),
        )
        simData = copy.deepcopy(SD)
        
    # Convert all layers to correct data format

    # extract the previous random state
    # state = data['state']
    # extract the previous simulation times

    simData.contactAgeGroupIndices = (
        np.digitize(
            np.array(t - simData.demography.birthDate),
            np.array(parameters.contactAgeGroupBreaks),
        )
        - 1
    )
    parameters.N = len(simData.si)

    # update the parameters
    R0 = simparams.iloc[j, 1].tolist()
    k = simparams.iloc[j, 2].tolist()
    parameters.R0 = R0
    parameters.k = k

    # configure the parameters
    parameters = configure(parameters)
    parameters.psi = getPsi(parameters)
    parameters.equiData = getEquilibrium(parameters)
    # parameters['moderateIntensityCount'], parameters['highIntensityCount'] = setIntensityCount(paramFileName)

    # add a simulation path
    # results = doRealizationSurveyCoveragePickle(params, simData, 1)
    # output = extractHostData(results)

    # transform the output to data frame
    df, simData = singleSimulationDALYCoverage(parameters, simData, 1)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"==> multiple_simulations finishing sim {i}: {total_time:.3f}s")
    return df, simData



def multiple_simulations_after_burnin(
    params: Parameters, pickleData, simparams, indices, i, burnInTime
) -> pd.DataFrame:
    print(f"==> multiple_simulations starting sim {i}")
    start_time = time.time()
    # copy the parameters
    parameters = copy.deepcopy(params)
    j = indices[i]
    # load the previous simulation results
    raw_data = pickleData[j]

    
    t = 0

    

    raw_data.demography.birthDate = raw_data.demography.birthDate - burnInTime
    raw_data.demography.deathDate = raw_data.demography.deathDate - burnInTime
    worms = Worms(total=raw_data.worms.total, female=raw_data.worms.female)
    demography = Demography(
        birthDate=raw_data.demography.birthDate,
        deathDate=raw_data.demography.deathDate,
    )
    simData = SDEquilibrium(
        si=raw_data.si,
        worms=worms,
        freeLiving=raw_data.freeLiving,
        demography=demography,
        contactAgeGroupIndices=raw_data.contactAgeGroupIndices,
        treatmentAgeGroupIndices=raw_data.treatmentAgeGroupIndices,
        sv=np.zeros(len(raw_data.si), dtype=int),
        attendanceRecord=[],
        ageAtChemo=[],
        adherenceFactorAtChemo=[],
        vaccCount=0,
        nChemo1=0,
        nChemo2=0,
        numSurvey=0,
        compliers=np.random.uniform(low=0, high=1, size=len(raw_data.si))
        > params.propNeverCompliers,
        adherenceFactors=np.random.uniform(low=0, high=1, size=len(raw_data.si)),
    )
    
    # Convert all layers to correct data format

    # extract the previous random state
    # state = data['state']
    # extract the previous simulation times

    simData.contactAgeGroupIndices = (
        np.digitize(
            np.array(t - simData.demography.birthDate),
            np.array(parameters.contactAgeGroupBreaks),
        )
        - 1
    )
    parameters.N = len(simData.si)

    # update the parameters
    R0 = simparams.iloc[j, 1].tolist()
    k = simparams.iloc[j, 2].tolist()
    parameters.R0 = R0
    parameters.k = k

    # configure the parameters
    parameters = configure(parameters)
    parameters.psi = getPsi(parameters)
    parameters.equiData = getEquilibrium(parameters)
    # parameters['moderateIntensityCount'], parameters['highIntensityCount'] = setIntensityCount(paramFileName)

    # add a simulation path
    # results = doRealizationSurveyCoveragePickle(params, simData, 1)
    # output = extractHostData(results)

    # transform the output to data frame
    df, simData = singleSimulationDALYCoverage(parameters, simData, 1)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"==> multiple_simulations finishing sim {i}: {total_time:.3f}s")
    return df, simData




def BurnInSimulations(
    params: Parameters, simparams, i
) -> pd.DataFrame:
    print(f"==> multiple_simulations starting sim {i}")
    start_time = time.time()
    #copy parameters
    parameters = copy.deepcopy(params)

    # update the parameters
    R0 = simparams.iloc[i, 1].tolist()
    k = simparams.iloc[i, 2].tolist()
    
    parameters.R0 = R0
    parameters.k = k
    # configure the parameters
    parameters = configure(parameters)
    parameters.psi = getPsi(parameters)
    parameters.equiData = getEquilibrium(parameters)
    
    # setup starting point form simulation
    simData = setupSD(parameters)
    simData.vaccCount=0
    simData.nChemo1=0
    simData.nChemo2=0
    simData.numSurvey=0


    df, simData = singleSimulationDALYCoverage(parameters, simData, 1)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"==> multiple_simulations finishing sim {i}: {total_time:.3f}s")
    return df, simData


def getLifeSpans(nSpans: int, params: Parameters) -> float:

    """
    This function draws the lifespans from the population survival curve.
    Parameters
    ----------
    nSpans: int
        number of drawings;
    params: Parameters object
        dataclass containing the parameter names and values;
    Returns
    -------
    array containing the lifespan drawings;
    """
    if params.hostAgeCumulDistr is None:
        raise ValueError("hostAgeCumulDistr is not set")
    u = np.random.uniform(low=0, high=1, size=nSpans) * np.max(params.hostAgeCumulDistr)
    # spans = np.array([np.argmax(v < params.hostAgeCumulDistr) for v in u])
    spans = np.argmax(
        params.hostAgeCumulDistr > u[:, None], axis=1
    )  # Should be faster?
    if params.muAges is None:
        raise ValueError("muAges not set")
    else:
        return params.muAges[spans]
