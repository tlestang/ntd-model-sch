import numpy as np
import pandas as pd
from scipy.optimize import bisect
import warnings
import pkg_resources
warnings.filterwarnings('ignore')
np.seterr(divide='ignore')

import sch_simulation.ParallelFuncs as ParallelFuncs

def readParam(fileName):

    '''
    This function extracts the parameter values stored
    in the input text files into a dictionary.

    Parameters
    ----------
    fileName: str
        name of the input text file;

    Returns
    -------
    params: dict
        dictionary containing the parameter names and values;
    '''

    DATA_PATH = pkg_resources.resource_filename('sch_simulation', 'data/')

    with open(DATA_PATH + fileName) as f:
        
        contents = f.readlines()

    params = []

    for content in contents:

        line = content.split('\t')

        if len(line) >= 2:

            try:
                
                line[1] = np.array([np.float(x) for x in line[1].split(' ')])

                if len(line[1]) == 1:
                    
                    line[1] = line[1][0]

            except:
                
                pass

            params.append(line[:2])

    params = dict(params)

    return params

def readParams(paramFileName, demogFileName='Demographies.txt', demogName='Default'):

    '''
    This function organizes the model parameters and
    the demography parameters into a unique dictionary.

    Parameters
    ----------
    paramFileName: str
        name of the input text file with the model parameters;

    demogFileName: str
        name of the input text file with the demography parameters;

    demogName: str
        subset of demography parameters to be extracted;

    Returns
    -------
    params: dict
        dictionary containing the parameter names and values;
    '''

    demographies = readParam(demogFileName)
    parameters = readParam(paramFileName)

    chemoTimings1 = np.array([parameters['treatStart1'] + x * parameters['treatInterval1']
    for x in range(np.int(parameters['nRounds1']))])

    chemoTimings2 = np.array([parameters['treatStart2'] + x * parameters['treatInterval2']
    for x in range(np.int(parameters['nRounds2']))])
    
    VaccineTimings = np.array([parameters['VaccTreatStart'] + x * parameters['treatIntervalVacc']
    for x in range(np.int(parameters['nRoundsVacc']))])

    params = {'numReps': np.int(parameters['repNum']),
              'maxTime': parameters['nYears'],
              'N': np.int(parameters['nHosts']),
              'R0': parameters['R0'],
              'lambda': parameters['lambda'],
              'v2': parameters['v2lambda'], # vacc par
              'gamma': parameters['gamma'],
              'k': parameters['k'],
              'sigma': parameters['sigma'],
              'v1':parameters['v1sigma'], # vacc par
              'LDecayRate': parameters['ReservoirDecayRate'],
              'DrugEfficacy': parameters['drugEff'],
              'contactAgeBreaks': parameters['contactAgeBreaks'],
              'contactRates': parameters['betaValues'],
              'v3': parameters['v3betaValues'],  # vacc par
              'rho': parameters['rhoValues'],
              'treatmentAgeBreaks': parameters['treatmentBreaks'],
              'VaccTreatmentBreaks': parameters['VaccTreatmentBreaks'], # vacc par
              'coverage1': parameters['coverage1'],
              'coverage2': parameters['coverage2'],
              'VaccCoverage':parameters['VaccCoverage'], #vacc par
              #'VaccEfficacy':parameters['vaccEff'], #vacc par
              'treatInterval1': parameters['treatInterval1'],
              'treatInterval2': parameters['treatInterval2'],
              'treatStart1': parameters['treatStart1'],
              'treatStart2': parameters['treatStart2'],
              'nRounds1': np.int(parameters['nRounds1']),
              'nRounds2': np.int(parameters['nRounds2']),
              'chemoTimings1': chemoTimings1,
              'chemoTimings2': chemoTimings2,
              'VaccineTimings' : VaccineTimings,
              'outTimings': parameters['outputEvents'],
              'propNeverCompliers': 0.0,
              'highBurdenBreaks': parameters['highBurdenBreaks'],
              'highBurdenValues': parameters['highBurdenValues'],
              'VaccDecayRate': parameters['VaccDecayRate'],
              'VaccTreatStart':parameters['VaccTreatStart'],
              'nRoundsVacc':parameters['nRoundsVacc'],
              'treatIntervalVacc':parameters['treatIntervalVacc'],
              'heavyThreshold':parameters['heavyThreshold'],
              'mediumThreshold':parameters['mediumThreshold'],
              
              'demogType': demogName,
              'hostMuData': demographies[demogName + '_hostMuData'],
              'muBreaks': np.append(0, demographies[demogName + '_upperBoundData']),
              'SR': [True if parameters['StochSR'] == 'TRUE' else False][0],
              'reproFuncName': parameters['reproFuncName'],
              'z': np.exp(-parameters['gamma']),
              'psi': 1.0,
              'k_epg': 0.87}

    return params

def configure(params):

    '''
    This function defines a number of additional parameters.

    Parameters
    ----------
    params: dict
        dictionary containing the initial parameter names and values;

    Returns
    -------
    params: dict
        dictionary containing the updated parameter names and values;
    '''

    # level of discretization for the drawing of lifespans
    dT = 0.1

    # definition of the reproduction function
    params['reproFunc'] = getattr(ParallelFuncs, params['reproFuncName'])

    # max age cutoff point
    params['maxHostAge'] = np.min([np.max(params['muBreaks']), np.max(params['contactAgeBreaks'])])

    # full range of ages
    params['muAges'] = np.arange(start=0, stop=np.max(params['muBreaks']), step=dT) + 0.5 * dT

    params['hostMu'] = params['hostMuData'][pd.cut(x=params['muAges'], bins=params['muBreaks'],
    labels=np.arange(start=0, stop=len(params['hostMuData']))).to_numpy()]

    # probability of surviving
    params['hostSurvivalCurve'] = np.exp(-np.cumsum(params['hostMu']) * dT)

    # the index for the last age group before the cutoff in this discretization
    maxAgeIndex = np.argmax([params['muAges'] > params['maxHostAge']]) - 1

    # cumulative probability of dying
    params['hostAgeCumulDistr'] = np.append(np.cumsum(dT * params['hostMu'] * np.append(1,
    params['hostSurvivalCurve'][:-1]))[:maxAgeIndex], 1)

    params['contactAgeGroupBreaks'] = np.append(params['contactAgeBreaks'][:-1], params['maxHostAge'])
    params['treatmentAgeGroupBreaks'] = np.append(params['treatmentAgeBreaks'][:-1], params['maxHostAge'] + dT)
    
    constructedVaccBreaks = np.sort(np.append(params['VaccTreatmentBreaks'], params['VaccTreatmentBreaks'] + 1))
    a = np.append(-dT, constructedVaccBreaks)
    params['VaccTreatmentAgeGroupBreaks'] = np.append(a, params['maxHostAge'] + dT)
    if params['outTimings'][-1] != params['maxTime']:
        params['outTimings'] = np.append(params['outTimings'], params['maxTime'])

    if params['reproFuncName'] == 'epgMonog':
        params['monogParams'] = ParallelFuncs.monogFertilityConfig(params)

    return params

def setupSD(params):

    '''
    This function sets up the simulation to initial conditions
    based on analytical equilibria.

    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;

    Returns
    -------
    SD: dict
        dictionary containing the equilibrium parameter settings;
    '''

    si = np.random.gamma(size=params['N'], scale=1 / params['k'], shape=params['k'])
    sv = np.zeros(params['N'], dtype=int)
    lifeSpans = getLifeSpans(params['N'], params)
    trialBirthDates = - lifeSpans * np.random.uniform(low=0, high=1, size=params['N'])
    trialDeathDates = trialBirthDates + lifeSpans

    communityBurnIn = 1000

    while np.min(trialDeathDates) < communityBurnIn:

        earlyDeath = np.where(trialDeathDates < communityBurnIn)[0]
        trialBirthDates[earlyDeath] = trialDeathDates[earlyDeath]
        trialDeathDates[earlyDeath] += getLifeSpans(len(earlyDeath), params)

    demography = {'birthDate': trialBirthDates - communityBurnIn, 'deathDate': trialDeathDates - communityBurnIn}

    contactAgeGroupIndices = pd.cut(x=-demography['birthDate'], bins=params['contactAgeGroupBreaks'],
    labels=np.arange(start=0, stop=len(params['contactAgeGroupBreaks']) - 1)).to_numpy()

    treatmentAgeGroupIndices = pd.cut(x=-demography['birthDate'], bins=params['treatmentAgeGroupBreaks'],
    labels=np.arange(start=0, stop=len(params['treatmentAgeGroupBreaks']) - 1)).to_numpy()

    meanBurdenIndex = pd.cut(x=-demography['birthDate'], bins=np.append(0, params['equiData']['ageValues']),
    labels=np.arange(start=0, stop=len(params['equiData']['ageValues']))).to_numpy()

    wTotal = np.random.poisson(lam=si * params['equiData']['stableProfile'][meanBurdenIndex] * 2, size=params['N'])

    worms = dict(total=wTotal, female=np.random.binomial(n=wTotal, p=0.5, size=params['N']))

    stableFreeLiving = params['equiData']['L_stable'] * 2
    
    VaccTreatmentAgeGroupIndices = pd.cut(x = -demography['birthDate'], bins = params['VaccTreatmentAgeGroupBreaks'],
    labels=np.arange(start=0, stop=len(params['VaccTreatmentAgeGroupBreaks']) - 1)).to_numpy()
    
    SD = {'si': si,
          'sv': sv,
          'worms': worms,
          'freeLiving': stableFreeLiving,
          'demography': demography,
          'contactAgeGroupIndices': contactAgeGroupIndices,
          'treatmentAgeGroupIndices': treatmentAgeGroupIndices,
          'VaccTreatmentAgeGroupIndices':VaccTreatmentAgeGroupIndices,
          'adherenceFactors': np.random.uniform(low=0, high=1, size=params['N']),
          'vaccinatedFactors': np.random.uniform(low=1, high=2, size=params['N']),
          'compliers': np.random.uniform(low=0, high=1, size=params['N']) > params['propNeverCompliers'],
          'attendanceRecord': [],
          'ageAtChemo': [],
          'adherenceFactorAtChemo': [],
          'vaccCount' :0
}

    return SD

def calcRates(params, SD):

    '''
    This function calculates the event rates; the events are
    new worms, worms death and vaccination recovery rates.

    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;

    SD: dict
        dictionary containing the equilibrium parameter values;

    Returns
    -------
    array of event rates;
    '''

    hostInfRates = SD['freeLiving'] * SD['si'] * params['contactRates'][SD['contactAgeGroupIndices']]
    deathRate = params['sigma'] * np.sum(SD['worms']['total'] * params['v1'][SD['sv']])
    hostVaccDecayRates = params['VaccDecayRate'][SD['sv']]
    return np.append(hostInfRates, hostVaccDecayRates, deathRate)


def calcRates2(params, SD):

    '''
    This function calculates the event rates; the events are
    new worms, worms death and vaccination recovery rates.
    Each of these types of events happen to individual hosts.

    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;

    SD: dict
        dictionary containing the equilibrium parameter values;

    Returns
    -------
    array of event rates;
    '''
        
    hostInfRates = SD['freeLiving'] * SD['si'] * params['contactRates'][SD['contactAgeGroupIndices']]
    deathRate = params['sigma'] * SD['worms']['total'] * params['v1'][SD['sv']]
    hostVaccDecayRates = params['VaccDecayRate'][SD['sv']]
    args = (hostInfRates, hostVaccDecayRates, deathRate)
    return np.concatenate(args)

def doEvent(rates, params, SD):

    '''
    This function enacts the event; the events are
    new worms, worms death and vaccine recoveries

    Parameters
    ----------
    rates: float
        array of event rates;

    SD: dict
        dictionary containing the initial equilibrium parameter values;

    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''

    # determine which event takes place; if it's 1 to N, it's a new worm, otherwise it's a worm death
    event = np.argmax(np.random.uniform(low=0, high=1, size=1) * np.sum(rates) < np.cumsum(rates))

    if event == len(rates) - 1: # worm death event

        deathIndex = np.argmax(np.random.uniform(low=0, high=1, size=1) * np.sum(SD['worms']['total'] * params['v1'][SD['sv']]) < np.cumsum(SD['worms']['total']* params['v1'][SD['sv']]))

        SD['worms']['total'][deathIndex] -= 1

        if np.random.uniform(low=0, high=1, size=1) < SD['worms']['female'][deathIndex] / SD['worms']['total'][deathIndex]:
            SD['worms']['female'][deathIndex] -= 1
    
    if event <= params['N']:
        if np.random.uniform(low=0, high=1, size=1) < params['v3'][SD['sv'][event]]:
            SD['worms']['total'][event] += 1
            if np.random.uniform(low=0, high=1, size=1) < 0.5:
                SD['worms']['female'][event] += 1
    elif event <= 2*params['N']:
        hostIndex = event - params['N']
        SD['sv'][hostIndex] = 0

    return SD



def doEvent2(rates, params, SD):

    '''
    This function enacts the event; the events are
    new worms, worms death and vaccine recoveries

    Parameters
    ----------
    rates: float
        array of event rates;

    SD: dict
        dictionary containing the initial equilibrium parameter values;

    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''
    
    event = np.argmax(np.random.uniform(low=0, high=1, size=1) * np.sum(rates) < np.cumsum(rates))
    eventType = ((event) // params['N']) + 1
    hostIndex = ((event) % params['N'])
    
    if eventType == 1:
        np.random.uniform(low=0, high=1, size=1) < params['v3'][SD['sv'][hostIndex]]
        SD['worms']['total'][hostIndex] += 1
        if np.random.uniform(low=0, high=1, size=1) < 0.5:
            SD['worms']['female'][hostIndex] += 1

    if eventType == 2:
        SD['sv'][hostIndex] = 0
        
    if eventType == 3:
        if np.random.uniform(low=0, high=1, size=1) < SD['worms']['female'][hostIndex]/SD['worms']['total'][hostIndex]:
            SD['worms']['female'][hostIndex] -= 1
        SD['worms']['total'][hostIndex] -= 1
        
    return SD
    
    
def doRegular(params, SD, t, dt):
    '''
    This function runs processes that happen regularly.
    These processes are reincarnating whicever hosts have recently died and
    updating the free living worm population

    Parameters
    ----------
    rates: float
        array of event rates;

    SD: dict
        dictionary containing the initial equilibrium parameter values;
    
    t:  int
        time point;
    
    dt: float
        time interval;
    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''
    
    
    SD = doDeath(params, SD, t)
    SD = doFreeLive(params, SD, dt)
    return SD
    
def doFreeLive(params, SD, dt):

    '''
    This function updates the freeliving population deterministically.

    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;

    SD: dict
        dictionary containing the initial equilibrium parameter values;

    dt: float
        time interval;

    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''

    # polygamous reproduction; female worms produce fertilised eggs only if there's at least one male worm around
    if params['reproFuncName'] == 'epgFertility' and params['SR']:
        productivefemaleworms = np.where(SD['worms']['total'] == SD['worms']['female'], 0, SD['worms']['female'])

    elif params['reproFuncName'] == 'epgFertility' and not params['SR']:
        productivefemaleworms = SD['worms']['female']

    # monogamous reproduction; only pairs of worms produce eggs
    elif params['reproFuncName'] == 'epgMonog':
        productivefemaleworms = np.minimum(SD['worms']['total'] - SD['worms']['female'], SD['worms']['female'])

    eggOutputPerHost = params['lambda'] * productivefemaleworms * np.exp(-SD['worms']['total'] * params['gamma']) * params['v2'][SD['sv']] # vaccine related fecundity
    eggsProdRate = 2 * params['psi'] * np.sum(eggOutputPerHost * params['rho'][SD['contactAgeGroupIndices']]) / params['N']
    expFactor = np.exp(-params['LDecayRate'] * dt)
    SD['freeLiving'] = SD['freeLiving'] * expFactor + eggsProdRate * (1 - expFactor) / params['LDecayRate']

    return SD

def doDeath(params, SD, t):

    '''
    Death and aging function.

    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;

    SD: dict
        dictionary containing the initial equilibrium parameter values;

    t: int
        time step;

    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''

    # identify the indices of the dead
    theDead = np.where(SD['demography']['deathDate'] < t)[0]

    if len(theDead) != 0:
        # they also need new force of infections (FOIs)
        SD['si'][theDead] = np.random.gamma(size=len(theDead), scale=1 / params['k'], shape=params['k'])
        SD['sv'][theDead] = 0

        # update the birth dates and death dates
        SD['demography']['birthDate'][theDead] = t - 0.001
        SD['demography']['deathDate'][theDead] = t + getLifeSpans(len(theDead), params)

        # kill all their worms
        SD['worms']['total'][theDead] = 0
        SD['worms']['female'][theDead] = 0

        # update the adherence factors
        SD['adherenceFactors'][theDead] = np.random.uniform(low=0, high=1, size=len(theDead))

        # assign the newly-born to either comply or not
        SD['compliers'][theDead] = np.random.uniform(low=0, high=1, size=len(theDead)) > params['propNeverCompliers']

    # update the contact age categories
    SD['contactAgeGroupIndices'] = pd.cut(x=t - SD['demography']['birthDate'], bins=params['contactAgeGroupBreaks'],
    labels=np.arange(0, len(params['contactAgeGroupBreaks']) - 1)).to_numpy()
    

    # update the treatment age categories
    SD['treatmentAgeGroupIndices'] = pd.cut(x=t - SD['demography']['birthDate'], bins=params['treatmentAgeGroupBreaks'],
    labels=np.arange(0, len(params['treatmentAgeGroupBreaks']) - 1)).to_numpy()
    SD['VaccTreatmentAgeGroupIndices'] = pd.cut(x=t - SD['demography']['birthDate'], bins=params['VaccTreatmentAgeGroupBreaks'],
    labels=np.arange(0, len(params['VaccTreatmentAgeGroupBreaks']) - 1)).to_numpy()
    return SD

def doChemo(params, SD, t, coverage):

    '''
    Chemoterapy function.

    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;

    SD: dict
        dictionary containing the initial equilibrium parameter values;

    t: int
        time step;

    coverage: array
        coverage fractions;

    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''

    # decide which individuals are treated, treatment is random
    attendance = np.random.uniform(low=0, high=1, size=params['N']) < coverage[SD['treatmentAgeGroupIndices']]

    # they're compliers and it's their turn
    toTreatNow = np.logical_and(attendance, SD['compliers'])

    # calculate the number of dead worms
    femaleToDie = np.random.binomial(size=np.sum(toTreatNow), n=SD['worms']['female'][toTreatNow],
    p=params['DrugEfficacy'])

    maleToDie = np.random.binomial(size=np.sum(toTreatNow), n=SD['worms']['total'][toTreatNow] -
    SD['worms']['female'][toTreatNow], p=params['DrugEfficacy'])

    SD['worms']['female'][toTreatNow] -= femaleToDie
    SD['worms']['total'][toTreatNow] -= (maleToDie + femaleToDie)

    # save actual attendance record and the age of each host when treated
    SD['attendanceRecord'].append(toTreatNow)
    SD['ageAtChemo'].append(t - SD['demography']['birthDate'])
    SD['adherenceFactorAtChemo'].append(SD['adherenceFactors'])

    return SD



def doVaccine(params, SD, t, VaccCoverage):
    '''
    Vaccine function.

    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;

    SD: dict
        dictionary containing the initial equilibrium parameter values;

    t: int
        time step;

    VaccCoverage: array
        coverage fractions;

    Returns
    -------
    SD: dict
        dictionary containing the updated equilibrium parameter values;
    '''
    temp  = ((SD['VaccTreatmentAgeGroupIndices'] + 1 ) // 2) - 1
    vaccinate = np.random.uniform(low=0, high=1, size=params['N']) < VaccCoverage[temp]
    
    indicesToVaccinate=[]
    for i in range(len(params['VaccTreatmentBreaks'])):
        indicesToVaccinate.append(1+i*2)
    Hosts4Vaccination = []
    for i in SD['VaccTreatmentAgeGroupIndices']:
        Hosts4Vaccination.append(i in indicesToVaccinate)
    
    vaccNow = np.logical_and(Hosts4Vaccination, vaccinate)
    SD['sv'][vaccNow] = 1
    SD['vaccCount'] += sum(Hosts4Vaccination) + sum(vaccinate)
    
    return SD
    
    
    
def getPsi(params):

    '''
    This function calculates the psi parameter.

    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;

    Returns
    -------
    value of the psi parameter;
    '''

    # higher resolution
    deltaT = 0.1

    # inteval-centered ages for the age intervals, midpoints from 0 to maxHostAge
    modelAges = np.arange(start=0, stop=params['maxHostAge'], step=deltaT) + 0.5 * deltaT

    # hostMu for the new age intervals
    hostMu = params['hostMuData'][pd.cut(x=modelAges, bins=params['muBreaks'], labels=np.arange(start=0,
    stop=len(params['hostMuData']))).to_numpy()]

    hostSurvivalCurve = np.exp(-np.cumsum(hostMu * deltaT))
    MeanLifespan = np.sum(hostSurvivalCurve[:len(modelAges)]) * deltaT

    # calculate the cumulative sum of host and worm death rates from which to calculate worm survival
    # intMeanWormDeathEvents = np.cumsum(hostMu + params['sigma']) * deltaT # commented out as it is not used

    modelAgeGroupCatIndex = pd.cut(x=modelAges, bins=params['contactAgeGroupBreaks'], labels=np.arange(start=0,
    stop=len(params['contactAgeGroupBreaks']) - 1)).to_numpy()

    betaAge = params['contactRates'][modelAgeGroupCatIndex]
    rhoAge = params['rho'][modelAgeGroupCatIndex]

    wSurvival = np.exp(-params['sigma'] * modelAges)

    B = np.array([np.sum(betaAge[: i] * np.flip(wSurvival[: i])) * deltaT for i in range(1, 1 + len(hostMu))])

    return params['R0'] * MeanLifespan * params['LDecayRate'] / (params['lambda'] * params['z'] *
    np.sum(rhoAge * hostSurvivalCurve * B) * deltaT)

def getLifeSpans(nSpans, params):

    '''
    This function draws the lifespans from the population survival curve.

    Parameters
    ----------
    nSpans: int
        number of drawings;

    params: dict
        dictionary containing the parameter names and values;

    Returns
    -------
    array containing the lifespan drawings;
    '''

    u = np.random.uniform(low=0, high=1, size=nSpans) * np.max(params['hostAgeCumulDistr'])
    spans = np.array([np.argmax(u[i] < params['hostAgeCumulDistr']) for i in range(nSpans)])

    return params['muAges'][spans]

def getEquilibrium(params):

    '''
    This function returns a dictionary containing the equilibrium worm burden
    with age and the reservoir value as well as the breakpoint reservoir value
    and other parameter settings.

    Parameters
    ----------
    params: dict
        dictionary containing the parameter names and values;

    Returns
    -------
    dictionary containing the equilibrium parameter settings;
    '''

    # higher resolution
    deltaT = 0.1

    # inteval-centered ages for the age intervals, midpoints from 0 to maxHostAge
    modelAges = np.arange(start=0, stop=params['maxHostAge'], step=deltaT) + 0.5 * deltaT

    # hostMu for the new age intervals
    hostMu = params['hostMuData'][pd.cut(x=modelAges, bins=params['muBreaks'], labels=np.arange(start=0,
    stop=len(params['hostMuData'])))]

    hostSurvivalCurve = np.exp(-np.cumsum(hostMu * deltaT))
    MeanLifespan = np.sum(hostSurvivalCurve[:len(modelAges)]) * deltaT

    modelAgeGroupCatIndex = pd.cut(x=modelAges, bins=params['contactAgeBreaks'], labels=np.arange(start=0,
    stop=len(params['contactAgeBreaks']) - 1)).to_numpy()

    betaAge = params['contactRates'][modelAgeGroupCatIndex]
    rhoAge = params['rho'][modelAgeGroupCatIndex]

    wSurvival = np.exp(-params['sigma'] * modelAges)

    # this variable times L is the equilibrium worm burden
    Q = np.array([np.sum(betaAge[: i] * np.flip(wSurvival[: i])) * deltaT for i in range(1, 1 + len(hostMu))])

    # converts L values into mean force of infection
    FOIMultiplier = np.sum(betaAge * hostSurvivalCurve) * deltaT / MeanLifespan

    # upper bound on L
    SRhoT = np.sum(hostSurvivalCurve * rhoAge) * deltaT
    R_power = 1 / (params['k'] + 1)
    L_hat = params['z'] * params['lambda'] * params['psi'] * SRhoT * params['k'] * (params['R0'] ** R_power - 1) / \
    (params['R0'] * MeanLifespan * params['LDecayRate'] * (1 - params['z']))

    # now evaluate the function K across a series of L values and find point near breakpoint;
    # L_minus is the value that gives an age-averaged worm burden of 1; negative growth should
    # exist somewhere below this
    L_minus = MeanLifespan / np.sum(Q * hostSurvivalCurve * deltaT)
    test_L = np.append(np.linspace(start=0, stop=L_minus, num=10), np.linspace(start=L_minus, stop=L_hat, num=20))

    def K_valueFunc(currentL, params):

        return params['psi'] * np.sum(params['reproFunc'](currentL * Q, params) * rhoAge * hostSurvivalCurve * deltaT) / \
        (MeanLifespan * params['LDecayRate']) - currentL

    K_values = np.vectorize(K_valueFunc)(currentL=test_L, params=params)

    # now find the maximum of K_values and use bisection to find critical Ls
    iMax = np.argmax(K_values)
    mid_L = test_L[iMax]

    if K_values[iMax] < 0:

        return dict(stableProfile=0 * Q,
                    ageValues=modelAges,
                    L_stable=0,
                    L_breakpoint=np.nan,
                    K_values=K_values,
                    L_values=test_L,
                    FOIMultiplier=FOIMultiplier)

    # find the top L
    L_stable = bisect(f=K_valueFunc, a=mid_L, b=4 * L_hat, args=(params))

    # find the unstable L
    L_break = test_L[1] / 50

    if K_valueFunc(L_break, params) < 0: # if it is less than zero at this point, find the zero
        L_break = bisect(f=K_valueFunc, a=L_break, b=mid_L, args=(params))

    stableProfile = L_stable * Q

    return dict(stableProfile=stableProfile, ageValues=modelAges,hostSurvival=hostSurvivalCurve,L_stable=L_stable,L_breakpoint=L_break,K_values=K_values,L_values=test_L,FOIMultiplier=FOIMultiplier)

def extractHostData(results):

    '''
    This function is used for processing results the raw simulation results.

    Parameters
    ----------
    results: list
        raw simulation output;

    Returns
    -------
    output: list
        processed simulation output;
    '''

    output = []

    for rep in range(len(results)):

        output.append(dict(
            wormsOverTime=np.array([results[rep][i]['worms']['total'] for i in range(len(results[0]) - 1)]).T,
            femaleWormsOverTime=np.array([results[rep][i]['worms']['female'] for i in range(len(results[0]) - 1)]).T,
            # freeLiving=np.array([results[rep][i]['freeLiving'] for i in range(len(results[0]) - 1)]),
            ages=np.array([results[rep][i]['time'] - results[rep][i]['hosts']['birthDate'] for i in range(len(results[0]) - 1)]).T,
            # adherenceFactors=np.array([results[rep][i]['adherenceFactors'] for i in range(len(results[0]) - 1)]).T,
            # compliers=np.array([results[rep][i]['compliers'] for i in range(len(results[0]) - 1)]).T,
            # totalPop=len(results[rep][0]['worms']['total']),
            timePoints=np.array([results[rep][i]['time'] for i in range(len(results[0]) - 1)]),
            # attendanceRecord=results[rep][-1]['attendanceRecord'],
            # ageAtChemo=results[rep][-1]['ageAtChemo'],
            # finalFreeLiving=results[rep][-2]['freeLiving'],
            # adherenceFactorAtChemo=results[rep][-1]['adherenceFactorAtChemo']
        ))

    return output

def getSetOfEggCounts(total, female, params, Unfertilized=False):

    '''
    This function returns a set of readings of egg counts from a vector of individuals,
    according to their reproductive biology.

    Parameters
    ----------
    total: int
        array of total worms;

    female: int
        array of female worms;

    params: dict
        dictionary containing the parameter names and values;

    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;

    Returns
    -------
    random set of egg count readings from a single sample;
    '''

    if Unfertilized:

        meanCount = female * params['lambda'] * params['z'] ** female

    else:

        eggProducers = np.where(total == female, 0, female)
        meanCount = eggProducers * params['lambda'] * params['z'] ** eggProducers

    return np.random.negative_binomial(size=len(meanCount), p=params['k_epg'] / (meanCount + params['k_epg']), n=params['k_epg'])

def getVillageMeanCountsByHost(villageList, timeIndex, params, nSamples=2, Unfertilized=False):

    '''
    This function returns the mean egg count across readings by host
    for a given time point and iteration.

    Parameters
    ----------
    villageList: dict
        processed simulation output for a given iteration;

    timeIndex: int
        selected time point index;

    nSamples: int
        number of samples;

    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;

    Returns
    -------
    array of mean egg counts;
    '''

    meanEggsByHost = getSetOfEggCounts(villageList['wormsOverTime'][:, timeIndex],
                                       villageList['femaleWormsOverTime'][:, timeIndex], params, Unfertilized) / nSamples

    for i in range(1, nSamples):

        meanEggsByHost += getSetOfEggCounts(villageList['wormsOverTime'][:, timeIndex],
                                            villageList['femaleWormsOverTime'][:, timeIndex], params, Unfertilized) / nSamples

    return meanEggsByHost

def getAgeCatSampledPrevByVillage(villageList, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    for a given time point and iteration.

    Parameters
    ----------
    villageList: dict
        processed simulation output for a given iteration;

    timeIndex: int
        selected time point index;

    ageBand: int
        array with age group boundaries;

    params: dict
        dictionary containing the parameter names and values;

    nSamples: int
        number of samples;

    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;

    villageSampleSize: int;
        village sample size fraction;

    Returns
    -------
    sampled worm prevalence;
    '''

    meanEggCounts = getVillageMeanCountsByHost(villageList, timeIndex, params, nSamples, Unfertilized)

    ageGroups = pd.cut(x=villageList['ages'][:, timeIndex], bins=np.append(-10, np.append(ageBand, 150)),
    labels=np.array([1, 2, 3])).to_numpy()

    currentAgeGroupMeanEggCounts = meanEggCounts[ageGroups == 2]

    if villageSampleSize < len(currentAgeGroupMeanEggCounts):
        mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=False)

    else:
        mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=True)

    return np.sum(nSamples * mySample > 0.9) / villageSampleSize

def getAgeCatSampledPrevByVillageAll(villageList, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    for a given time point and iteration.

    Parameters
    ----------
    villageList: dict
        processed simulation output for a given iteration;

    timeIndex: int
        selected time point index;

    ageBand: int
        array with age group boundaries;

    params: dict
        dictionary containing the parameter names and values;

    nSamples: int
        number of samples;

    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;

    villageSampleSize: int;
        village sample size fraction;

    Returns
    -------
    sampled worm prevalence;
    '''

    meanEggCounts = getVillageMeanCountsByHost(villageList, timeIndex, params, nSamples, Unfertilized)

    ageGroups = pd.cut(x=villageList['ages'][:, timeIndex], bins=np.append(-10, np.append(ageBand, 150)),
    labels=np.array([1, 2, 3])).to_numpy()

    currentAgeGroupMeanEggCounts = meanEggCounts[ageGroups == 2]

    if villageSampleSize < len(currentAgeGroupMeanEggCounts):
        mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=False)

    else:
        mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=True)

    infected = np.sum(nSamples * mySample > 0.9) / villageSampleSize
    #low = np.sum((mySample >= 1) & (mySample < params['mediumThreshold']))) / villageSampleSize
    medium = np.sum((mySample >= params['mediumThreshold']) & (mySample <= params['heavyThreshold'])) / villageSampleSize
    heavy = np.sum(mySample > params['heavyThreshold']) / villageSampleSize

    low = infected - (medium + heavy)

    return infected, low, medium, heavy


def getAgeCatSampledPrevHeavyBurdenByVillage(villageList, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    for a given time point and iteration.

    Parameters
    ----------
    villageList: dict
        processed simulation output for a given iteration;

    timeIndex: int
        selected time point index;

    ageBand: int
        array with age group boundaries;

    params: dict
        dictionary containing the parameter names and values;

    nSamples: int
        number of samples;

    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;

    villageSampleSize: int;
        village sample size fraction;

    Returns
    -------
    sampled worm prevalence;
    '''

    meanEggCounts = getVillageMeanCountsByHost(villageList, timeIndex, params, nSamples, Unfertilized)

    ageGroups = pd.cut(x=villageList['ages'][:, timeIndex], bins=np.append(-10, np.append(ageBand, 150)),
    labels=np.array([1, 2, 3])).to_numpy()

    currentAgeGroupMeanEggCounts = meanEggCounts[ageGroups == 2]

    if villageSampleSize < len(currentAgeGroupMeanEggCounts):
        mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=False)

    else:
        mySample = np.random.choice(a=currentAgeGroupMeanEggCounts, size=villageSampleSize, replace=True)

    return np.sum(mySample > 16) / villageSampleSize


def getSampledDetectedPrevByVillageAll(hostData, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    at a given time point across all iterations.

    Parameters
    ----------
    hostData: dict
        processed simulation output;

    timeIndex: int
        selected time point index;

    ageBand: int
        array with age group boundaries;

    params: dict
        dictionary containing the parameter names and values;

    nSamples: int
        number of samples;

    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;

    villageSampleSize: int;
        village sample size fraction;

    Returns
    -------
    sampled worm prevalence;
    '''

    return np.array([getAgeCatSampledPrevByVillageAll(villageList, timeIndex, ageBand, params,
    nSamples, Unfertilized, villageSampleSize) for villageList in hostData])

def getBurdens(hostData, params, numReps, ageBand, nSamples=2, Unfertilized=False, villageSampleSize=100):
   
    results= np.empty((0,numReps))
    low_results = np.empty((0,numReps))
    medium_results = np.empty((0,numReps))
    heavy_results = np.empty((0,numReps))

    for t in range(len(hostData[0]['timePoints'])): #loop over time points
    # calculate burdens using the same sample
        newrow = getSampledDetectedPrevByVillageAll(hostData, t, ageBand, params, nSamples, Unfertilized, villageSampleSize)
        newrowinfected = newrow[:,0]
        newrowlow = newrow[:,1]
        newrowmedium = newrow[:,2]
        newrowheavy = newrow[:,3]
        # append row
        results = np.vstack([results, newrowinfected])
        low_results = np.vstack([low_results, newrowlow])
        medium_results = np.vstack([medium_results, newrowmedium])
        heavy_results = np.vstack([heavy_results, newrowheavy])

    # calculate proportion across number of repetitions
    prevalence = np.sum(results, axis = 1) / numReps
    low_prevalence = np.sum(low_results, axis = 1) / numReps
    medium_prevalence = np.sum(medium_results, axis = 1) / numReps
    heavy_prevalence = np.sum(heavy_results, axis = 1) / numReps

    return prevalence,low_prevalence,medium_prevalence,heavy_prevalence

def getSampledDetectedPrevByVillage(hostData, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    at a given time point across all iterations.

    Parameters
    ----------
    hostData: dict
        processed simulation output;

    timeIndex: int
        selected time point index;

    ageBand: int
        array with age group boundaries;

    params: dict
        dictionary containing the parameter names and values;

    nSamples: int
        number of samples;

    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;

    villageSampleSize: int;
        village sample size fraction;

    Returns
    -------
    sampled worm prevalence;
    '''

    return np.array([getAgeCatSampledPrevByVillage(villageList, timeIndex, ageBand, params,
    nSamples, Unfertilized, villageSampleSize) for villageList in hostData])

def getSampledDetectedPrevHeavyBurdenByVillage(hostData, timeIndex, ageBand, params, nSamples=2, Unfertilized=False,
    villageSampleSize=100):

    '''
    This function provides sampled, age-cat worm prevalence
    at a given time point across all iterations.

    Parameters
    ----------
    hostData: dict
        processed simulation output;

    timeIndex: int
        selected time point index;

    ageBand: int
        array with age group boundaries;

    params: dict
        dictionary containing the parameter names and values;

    nSamples: int
        number of samples;

    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;

    villageSampleSize: int;
        village sample size fraction;

    Returns
    -------
    sampled worm prevalence;
    '''

    return np.array([getAgeCatSampledPrevHeavyBurdenByVillage(villageList, timeIndex, ageBand, params,
    nSamples, Unfertilized, villageSampleSize) for villageList in hostData])

def getPrevalence(hostData, params, numReps, nSamples=2, Unfertilized=False, villageSampleSize=100):

    '''
    This function provides the average SAC and adult prevalence at each time point,
    where the average is calculated across all iterations.

    Parameters
    ----------
    hostData: dict
        processed simulation output;

    params: dict
        dictionary containing the parameter names and values;

    numReps: int
        number of simulations;

    nSamples: int
        number of samples;

    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;

    villageSampleSize: int;
        village sample size fraction;

    Returns
    -------
    data frame with SAC and adult prevalence at each time point;
    '''

    sac_results = np.array([getSampledDetectedPrevByVillage(hostData, t, np.array([5, 15]), params, nSamples,
    Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    adult_results = np.array([getSampledDetectedPrevByVillage(hostData, t, np.array([16, 80]), params, nSamples,
    Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    sac_heavy_results = np.array([getSampledDetectedPrevHeavyBurdenByVillage(hostData, t, np.array([5, 15]), params,
    nSamples, Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    adult_heavy_results = np.array([getSampledDetectedPrevHeavyBurdenByVillage(hostData, t, np.array([16, 80]), params,
    nSamples, Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    sac_prevalence = np.sum(sac_results, axis=1) / numReps
    adult_prevalence = np.sum(adult_results, axis=1) / numReps

    sac_heavy_prevalence = np.sum(sac_heavy_results, axis=1) / numReps
    adult_heavy_prevalence = np.sum(adult_heavy_results, axis=1) / numReps

    df = pd.DataFrame({'Time': hostData[0]['timePoints'],
                       'SAC Prevalence': sac_prevalence,
                       'Adult Prevalence': adult_prevalence,
                       'SAC Heavy Intensity Prevalence': sac_heavy_prevalence,
                       'Adult Heavy Intensity Prevalence': adult_heavy_prevalence})

    df = df[(df['Time'] >= 50) & (df['Time'] <= 64)]
    df['Time'] = df['Time'] - 50

    return df

def getPrevalenceDALYs(hostData, params, numReps, nSamples=2, Unfertilized=False, villageSampleSize=100):

    '''
    This function provides the average SAC and adult prevalence at each time point,
    where the average is calculated across all iterations.

    Parameters
    ----------
    hostData: dict
        processed simulation output;

    params: dict
        dictionary containing the parameter names and values;

    numReps: int
        number of simulations;

    nSamples: int
        number of samples;

    Unfertilized: bool
        True / False flag for whether unfertilized worms generate eggs;

    villageSampleSize: int;
        village sample size fraction;

    Returns
    -------
    data frame with SAC and adult prevalence at each time point;
    '''
    
    all_results = np.array([getSampledDetectedPrevByVillage(hostData, t, np.array([0, 80]), params, nSamples,
    Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    ufour_results = np.array([getSampledDetectedPrevByVillage(hostData, t, np.array([0, 4]), params, nSamples,
    Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    adult_results = np.array([getSampledDetectedPrevByVillage(hostData, t, np.array([5, 80]), params, nSamples,
    Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    all_heavy_results = np.array([getSampledDetectedPrevHeavyBurdenByVillage(hostData, t, np.array([0, 80]), params,
    nSamples, Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    ufour_heavy_results = np.array([getSampledDetectedPrevHeavyBurdenByVillage(hostData, t, np.array([0, 4]), params,
    nSamples, Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    adult_heavy_results = np.array([getSampledDetectedPrevHeavyBurdenByVillage(hostData, t, np.array([5, 80]), params,
    nSamples, Unfertilized, villageSampleSize) for t in range(len(hostData[0]['timePoints']))])

    all_prevalence = np.sum(all_results, axis=1) / numReps
    ufour_prevalence = np.sum(ufour_results, axis=1) / numReps
    adult_prevalence = np.sum(adult_results, axis=1) / numReps

    all_heavy_prevalence = np.sum(all_heavy_results, axis=1) / numReps
    ufour_heavy_prevalence = np.sum(ufour_heavy_results, axis=1) / numReps
    adult_heavy_prevalence = np.sum(adult_heavy_results, axis=1) / numReps

    #under 4s
    ufour_prevalence_check, ufour_low_prevalence_check, ufour_medium_prevalence_check, ufour_heavy_prevalence_check = getBurdens(hostData, params, numReps, np.array([0, 4]), nSamples=2, Unfertilized=False, villageSampleSize=100)

    # adults
    adult_prevalence_check, adult_low_prevalence_check, adult_medium_prevalence_check, adult_heavy_prevalence_check = getBurdens(hostData, params, numReps, np.array([5, 80]), nSamples=2, Unfertilized=False, villageSampleSize=100)

    #all individuals 
    all_prevalence_check, all_low_prevalence_check, all_medium_prevalence_check, all_heavy_prevalence_check = getBurdens(hostData, params, numReps, np.array([0, 80]), nSamples=2, Unfertilized=False, villageSampleSize=100)


    df = pd.DataFrame({'Time': hostData[0]['timePoints'],
                        'Prevalence old': all_prevalence,
                       'Prevalence': all_prevalence_check,
                        'Low Intensity Prevalence': all_low_prevalence_check,
                        'Medium Intensity Prevalence': all_medium_prevalence_check,
                        'Heavy Intensity Prevalence': all_heavy_prevalence_check,
                        'Heavy Intensity Prevalence old': all_heavy_prevalence,
                       'Under four old': ufour_prevalence,
                       'Under four' : ufour_prevalence_check,
                        'Under four Low Intensity Prevalence': ufour_low_prevalence_check,
                       'Under four Medium Intensity Prevalence': ufour_medium_prevalence_check,
                       'Under four Heavy Intensity Prevalence': ufour_heavy_prevalence_check,
                       'Under four Heavy Intensity Prevalence old ': ufour_heavy_prevalence,
                        'Adult Prevalence old': adult_prevalence,
                        'Adult Prevalence': adult_prevalence_check,
                        'Adult Heavy Intensity Prevalence old': adult_heavy_prevalence,
                        'Adult Low Intensity Prevalence': adult_low_prevalence_check,
                        'Adult Medium Intensity Prevalence': adult_medium_prevalence_check,
                        'Adult Heavy Intensity Prevalence': adult_heavy_prevalence_check})


    df = df[(df['Time'] >= 50) & (df['Time'] <= 64)]
    df['Time'] = df['Time'] - 50

    return df
