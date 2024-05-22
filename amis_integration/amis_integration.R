library(reticulate)
library(AMISforInfectiousDiseases)

get_venv <- function() {
  result <- system2(
    command = "pipenv",
    args = "--venv",
    stdout = TRUE,
    stderr = FALSE
  )
  if (length(result) == 0) {
    # See the main README under How to run for instructions
    # on how to setup the virtual environment
    stop("No Python virtual enviroment found,
      install amis_intergration before running this script
      and ensure the working directory is inside the project")
  }
  return(result)
}

use_virtualenv(get_venv())
importlib <- import("importlib")
sch_simulation <- import("amis_integration")
importlib$reload(sch_simulation)

fixed_parameters <- sch_simulation$FixedParameters(
  number_hosts = 10,
  # no intervention
  coverage_file_name = "mansoni_coverage_scenario_0.xlsx",
  demography_name = "UgandaRural",
  survey_type = "KK2",
  parameter_file_name = "mansoni_params.txt",
  coverage_text_file_storage_name = "Man_MDA_vacc.txt"
)

# Example prevalence map, with two locations, both with prevalence of 0.5
prevalence_map <- matrix(c(0.5, 0.5), ncol = 1)

transmission_model <- function(seeds, params, n_tims = length(prevalence_map)) {
  output <- sch_simulation$run_model_with_parameters(
    seeds, params, fixed_parameters
  )
  print("Output:")
  print(output)
  return(output)
}

#' the "dprior" function
#' Note the second parameter _must_ be called log
#' Unclear what this function does
density_function <- function(parameters, log) {
  return(0.5)
}

#' The "rprior" function that returns a matrix whose columns are the parameters
#' and each row is a sample
rnd_function <- function(num_samples) {
  return(matrix(c(3, 0.04), ncol = 2, nrow = num_samples, byrow = TRUE))
}

prior <- list("dprior" = density_function, "rprior" = rnd_function)


amis_output <- amis(
  prevalence_map,
  transmission_model,
  prior
)

# Currently errors - I think because I
# don't know where the weights need to be set
# "No weight on any particles for locations in the active set."
print(amis_output)
