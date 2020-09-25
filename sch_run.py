from sch_simulation.helsim_RUN_KK import SCH_Simulation

print ("hello 1")
df = SCH_Simulation(paramFileName='SCH-high_adult_burden.txt', demogName='UgandaRural', numReps=10)
print ("hello 2")

df.to_json('sch_results.json')
print ("hello 3")

df.plot(x='Time', y=['SAC Prevalence', 'Adult Prevalence'])
print ("hello 4")

df.plot(x='Time', y=['SAC Heavy Intensity Prevalence', 'Adult Heavy Intensity Prevalence'])
print ("hello 5")
