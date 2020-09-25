from sch_simulation.helsim_RUN_KK import SCH_Simulation

df = SCH_Simulation(paramFileName='SCH-high_adult_burden.txt', demogName='UgandaRural', numReps=10)

df.to_json('sch_results.json')

df.plot(x='Time', y=['SAC Prevalence', 'Adult Prevalence'])

df.plot(x='Time', y=['SAC Heavy Intensity Prevalence', 'Adult Heavy Intensity Prevalence'])
