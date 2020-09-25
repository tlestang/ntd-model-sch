# SCH Simulation Model

To run the SCH simulation model, import the `SCH_Simulation()` function from the `helsim_RUN_KK` module in the `sch_simulation` package.

The `SCH_Simulation()` function requires the following inputs:

- `paramFileName`: name of the input file with the model parameters. The available files are listed in the following table.

| paramFileName |
| :--- | 
| SCH-high_adult_burden.txt | 
| SCH-low_adult_burden.txt  | 

- `demogName`: name of the demography. The available demographies are listed in the following table.

| demogName | 
| :--- | 
| Default | 
| WHOGeneric  | 
| UgandaRural | 
| KenyaKDHS  | 
| Flat | 

- `numReps`: number of simulations. If not provided, the number of simulations is extracted from the parameters file.

The `SCH_Simulation()` function returns a data frame with the following columns: `time`, `SAC Prevalence`, 
`Adult Prevalence`, `SAC Heavy Intensity Prevalence` and `Adult Heavy Intensity Prevalence`.

The output data frame can be exported in several different formats; see `sch_results.json` for an example of the results in JSON format.

See also `sch_run.py` for an example of how to use the `SCH_Simulation()` function.

### How to run

- Install [pipenv](https://drive.google.com/drive/folders/1Or6lUkymYd_p031xKGZLcnTV4GYf-oYb) according to the instructions for your OS, then `cd` to the project directory and run:

```
    $ pipenv install . # sets up per-project python environment ('env')
    $ pipenv shell # starts a per-project shell using that env
    (ntd-model-trachoma) $ python trachoma_run.py # runs the model
```
