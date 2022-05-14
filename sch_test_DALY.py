from sch_simulation.helsim_RUN_KK import SCH_Simulation_DALY


# Schisto (mansoni)
df0 = SCH_Simulation_DALY(paramFileName='SCH_params/mansoni_scenario_1.txt', demogName='UgandaRural', numReps=1)
df0.to_csv('sch_test.csv')

df0 = SCH_Simulation_DALY(paramFileName='SCH_params/mansoni_scenario_2.txt', demogName='UgandaRural', numReps=1)
df0.to_csv('sch_test_3.csv')

# Hookworm
df0 = SCH_Simulation_DALY(paramFileName='STH_params/hookworm_scenario_1.txt', demogName='UgandaRural', numReps=1)
df0.to_csv('sth_hookworm_test.csv')

# Ascaris
df0 = SCH_Simulation_DALY(paramFileName='STH_params/ascaris_scenario_1.txt', demogName='UgandaRural', numReps=1)
df0.to_csv('sth_ascaris_test.csv')

# Trichuris
df0 = SCH_Simulation_DALY(paramFileName='STH_params/trichuris_scenario_1.txt', demogName='UgandaRural', numReps=1)
df0.to_csv('sth_trichuris_test.csv')
