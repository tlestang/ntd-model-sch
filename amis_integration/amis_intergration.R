library(reticulate)
library(AMISforInfectiousDiseases)

# TODO: Get the virtual enviroment programatically
use_virtualenv("/home/thomas/.local/share/virtualenvs/ntd-model-sch-an6K7Edb")
importlib <- import("importlib")
sch_simulation <- import("amis_integration")
importlib$reload(sch_simulation)

# Example prevalence map, with two locations, both with prevalence of 0.5
prevalence_map <- matrix(c(0.5, 0.5), ncol = 1)

transmission_model <- function(seeds, params, n_tims = length(prevalence_map)) {
  output <- sch_simulation$run_model_with_parameters(seeds, params)
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
