library(reticulate)
library(AMISforInfectiousDiseases)

# Example prevalence map, with two locations, both with prevalence of 0.5
prevalence_map <- matrix(c(0.5, 0.5), ncol = 1)
print(prevalence_map)

transmission_model <- function(seeds, params, n_tims = length(prevalence_map)) {
  print("Input params: ")
  print(params)
  # With this value, AMIS says two locations are below the target
  # TODO: Why is the value 1, and not 0.5 as specified in the prevalence map?
  output <- matrix(c(1), ncol = 1, nrow = nrow(params))
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
  return(matrix(c(1, 2), ncol = 2, nrow = num_samples))
}

prior <- list("dprior" = density_function, "rprior" = rnd_function)


amis_output <- amis(
  prevalence_map,
  transmission_model,
  prior
)

# Currently errors - I think because it is not expecting to
# find a fit on the first iteration, but possibly because I
# don't know where the weights need to be set

print(amis_output)
