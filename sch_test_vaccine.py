from sch_simulation.helsim_RUN_KK import SCH_Simulation


# simulation with no vaccine (specified as coverage = 0)
df0 = SCH_Simulation(paramFileName='SCH_paramsKKvacc0.txt', demogName='UgandaRural', numReps=100)
df0.to_csv('sch_vaccine_branch_no_vaccine.csv')


# simulation with vaccine (using same parameters as R)
df = SCH_Simulation(paramFileName='SCH_paramsKKvacc.txt', demogName='UgandaRural', numReps=100)
df.to_csv('sch_vaccine_branch_vaccine.csv')


