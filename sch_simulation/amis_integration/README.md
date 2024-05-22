AMIS Integration
================

Provides the relevant wrapper code to be able to call the SCH Simulation from 
the AMIS loop to compute fitted parameters. 

# Usage

## Prerequistes

 * R
 * Renv 

## Installation

Make sure you've set up the Python virtual environment for the
sch_simulation package (see main [README](../../README.md)).

1. Launch a R session `R`
2. Create a R virtual environment `renv::init()`
3. Install dependencies `renv::restore()`

## Running

```bash
$ Rscript amis_integration.R
```