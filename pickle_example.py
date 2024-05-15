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
saveResults = True

# path to pickle file
InSimFilePath = 'SD1.pickle'

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
coverageFileName = 'Coverage_template_3.xlsx'
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

numSims = 200
print( f'Running {numSims} simulations on {num_cores} cores' )


# randomly pick indices for number of simulations
#indices = np.random.choice(a=range(200), size = numSims, replace=False)

# pick parameters and saved populations in order
indices = range(200)
start_time = time.time()

# run simulations in parallel
# run simulations in parallel
res = Parallel(n_jobs=num_cores)(
         delayed(multiple_simulations)(params, pickleData, simparams, indices, i) for i in range(numSims))
results=[item[0] for item in res]
simData:list[SDEquilibrium]=[item[1] for item in res]
#results = Parallel(n_jobs=1)(
 #        delayed(multiple_simulations)(params, pickleData, simparams, indices, i) for i in range(numSims))
newOutputSimDataFilePath = 'SDPickles.pkl'
pickle.dump( simData, open( newOutputSimDataFilePath, 'wb' ) )
    
end_time = time.time()

print(end_time - start_time)

if saveResults:
    for i in range(len(results)):
        df = results[i]
        df.to_csv(f'sch_simulation/data/simResults/resultsc_{i:03}.csv', index=False)
