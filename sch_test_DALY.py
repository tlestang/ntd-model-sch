from sch_simulation.helsim_RUN_KK import SCH_Simulation_DALY


# simulation with no vaccine (specified as coverage = 0)
df0 = SCH_Simulation_DALY(paramFileName='test.txt', demogName='UgandaRural', numReps=1)
df0.to_csv('SCH_test.csv')





