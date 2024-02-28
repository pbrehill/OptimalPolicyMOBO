library(tidyverse)
library(policytree)
library(rpart)

evaluate_tree <- function(X, gamma1, gamma2, g1_weight, search_depth = 2, se = TRUE, depth = NULL, bs) {
  get_utility <- function(predictions, gammax) {
  gamma_flip <- gammax %>% t() %>% as.data.frame.matrix
  utilities <- map2(gamma_flip, predictions, ~.x[.y])
  surrogate_util <- mean(unlist(utilities))
  oracle_util <- gammax %>% apply(1, max) %>% mean()
  return(list(mean_utility = surrogate_util, oracle_utility = oracle_util, regret = oracle_util - surrogate_util))
}

honest_pt <- function(X, gamma1, gamma2, g1_weight, search_depth, se, depth, bs) {
  # Add in mix of gammas for fitting
  gamma <- (gamma1 * g1_weight + 0.000000001) + ((1.0 - g1_weight + 0.000000001) * gamma2)
#   write_csv(gamma, paste0("dr_scores/gamma", round(g1_weight, 3), ".csv")


  if (search_depth == depth) {
    output <- capture.output({
        ht1 = policy_tree(X, gamma, depth = depth)
    })

  } else {
    output <- capture.output ({
        ht1 = hybrid_policy_tree(X, gamma, depth = depth, search.depth = search_depth)
        })
     }
  sink(NULL)

  # Predict on out of sample
  ht1_pred <- predict(ht1, X)

  g1_util <- get_utility(ht1_pred, gamma1)[["mean_utility"]]
  g2_util <- get_utility(ht1_pred, gamma2)[["mean_utility"]]

  c(g1_util, g2_util)
}

  utility_point <- honest_pt(X, gamma1, gamma2, g1_weight, search_depth, depth=depth)
  if (se) {
    sd <- map(1:bs, function (a) {
          dt = sort(sample.int(nrow(X), nrow(X), replace = TRUE))
          honest_pt(X[dt,], gamma1[dt,], gamma2[dt,], g1_weight, search_depth, depth=depth)
  }
  )

  se1 <- sd(map_dbl(sd, ~.x[1])) / sqrt(bs)
  se2 <- sd(map_dbl(sd, ~.x[2])) / sqrt(bs)
  } else {
  se1 <- 0
  se2 <- 0
  }

  return(c(utility_point[1], utility_point[2], se1, se2))
}

get_oracles <- function(gamma1, gamma2, g1_weight) {
  # Get the weighted utility matrix
  gamma <- (gamma1 * g1_weight + 0.000000001) + ((1.0 - g1_weight + 0.000000001) * gamma2)

  # Find the maximum
  maxes <- max.col(gamma)

  # Calculate seperate utilities for the treatment
  g1_total <- c()
  for (i in 1:length(maxes)) c(g1_total , gamma1[i, maxes[i]])

  g2_total <- c()
  for (i in 1:length(maxes)) c(g2_total , gamma2[i, maxes[i]])

  return(list(g1 = g1_total, g2 = g2_total))

}



