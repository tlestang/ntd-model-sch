from sch_simulation.helsim_RUN_KK import *
from sch_simulation.helsim_FUNC_KK import *
import pickle
import time
import multiprocessing

'''
File to load in a pickle file and associated parameters file and then
run forward in time 23 years and give back results
Can store results as a set of csv's if saveResults = True
'''

# flag for saving results or not
saveResults = False

# path to pickle file
InSimFilePath = 'Man_AGO02049.p'

#pickleData = pickle.loads( cloudModule.get_blob( InSimFilePath ) ) if useCloudStorage else pickle.load(open(InSimFilePath, 'rb'))

# load pickle file
pickleData =  pickle.load(open(InSimFilePath, 'rb'))

# path to 200 parameters file
RkFilePath = 'Input_Rk_Man_AGO02049.csv'
# read in parameters
simparams = pd.read_csv(RkFilePath)
simparams.columns = [s.replace(' ', '') for s in simparams.columns]

# define the lists of random seeds, R0 and k
seed = simparams.iloc[:, 0].tolist()
R0 = simparams.iloc[:, 1].tolist()
k = simparams.iloc[:, 2].tolist()

# path to coverage file
coverageFileName = 'Coverage_template.xlsx'
demogName = 'Default'

# file name to store munged coverage information in 
coverageTextFileStorageName = 'Man_AGO02049_MDA_vacc.txt'

# standard parameter file path (in sch_simulation/data folder)
paramFileName = 'sch_example.txt'

# parse coverage file
cov = parse_coverage_input(coverageFileName,
     coverageTextFileStorageName)

# initialize the parameters
params = loadParameters(paramFileName, demogName)
# add coverage data to parameters file
params = readCoverageFile(coverageTextFileStorageName, params)

# count number of processors
num_cores = multiprocessing.cpu_count()

# number of simulations to run
numSims = num_cores * 3
print( f'Running {numSims} simulations on {num_cores} cores' )
#numSims = 2

# randomly pick indices for number of simulations
indices = np.random.choice(a=range(200), size = numSims, replace=False)

start_time = time.time()

# run simulations in parallel
results = Parallel(n_jobs=num_cores)(
        delayed(multiple_simulations)(params, pickleData, simparams, i) for i in range(numSims))
    
end_time = time.time()

print(end_time - start_time)

if saveResults:
    for i in range(len(results)):
        df = results[i]
        df.to_csv(f'sch_simulation/data/simResults/results_{i:03}.csv', index=False)
