from sch_simulation.helsim_RUN_KK import SCH_Simulation_DALY


# simulation with no vaccine (specified as coverage = 0)
df0 = SCH_Simulation_DALY(paramFileName='SCH-high_adult_burden.txt', demogName='UgandaRural', numReps=1)
df0.to_csv('SCH-high_adult_burden.csv')


# simulation with no vaccine (specified as coverage = 0)
#df0 = SCH_Simulation_DALY(paramFileName='SCH-low_adult_burden.txt', demogName='UgandaRural', numReps=1)
#df0.to_csv('SCH-low_adult_burden.csv')


