library(tidyverse)
library(policytree)

evaluate_tree <- function(X, gamma) {
  get_utility <- function(predictions, gamma) {
  utilities <- c()
  for (i in 1:nrow(gamma)) utilities <- c(utilities, gamma[i, predictions[i]])
  surrogate_util <- mean(unlist(utilities))
  oracle_util <- gamma %>% apply(1, max) %>% mean()
  return(list(mean_utility = surrogate_util, oracle_utility = oracle_util, regret = oracle_util - surrogate_util))
}

honest_pt <- function(X, gamma) {
  get_utility(
    predict(
      hybrid_policy_tree(X, gamma, depth = 3), X),
    gamma)[["mean_utility"]]
}

  utility_point <- honest_pt(X, gamma)
  sd <- map_dbl(1:50, function (a) {
          dt = sort(sample.int(nrow(X), nrow(X)*.5, replace = TRUE))
          honest_pt(X[dt,], gamma[dt,])
  }
  ) %>% sd()
  se <- sd / sqrt(nrow(X))
  return(c(utility_point, se))
}