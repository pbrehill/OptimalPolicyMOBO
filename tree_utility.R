library(tidyverse)
library(policytree)
library(rpart)

evaluate_tree <- function(X, gamma1, gamma2, g1_weight, search_depth = 2) {
  get_utility <- function(predictions, gamma) {
  utilities <- c()
  for (i in 1:nrow(gamma)) utilities <- c(utilities, gamma[i, predictions[i]])
  surrogate_util <- mean(unlist(utilities))
  oracle_util <- gamma %>% apply(1, max) %>% mean()
  return(list(mean_utility = surrogate_util, oracle_utility = oracle_util, regret = oracle_util - surrogate_util))
}

honest_pt <- function(X, gamma1, gamma2, g1_weight, search_depth) {
  # Add in mix of gammas for fitting
  gamma = gamma1 - gamma2 # Benefits - cost
  dt = sort(sample.int(nrow(X), nrow(X)*.5, replace = FALSE))

  if (search_depth == 3) {
    ht1 = policy_tree(X[dt,], gamma[dt,], depth = 3)
    ht2 = policy_tree(X[-dt,], gamma[-dt,], depth = 3)

  } else {
    ht1 = hybrid_policy_tree(X[dt,], gamma[dt,], depth = 3, search.depth = search_depth)
    ht2 = hybrid_policy_tree(X[-dt,], gamma[-dt,], depth = 3, search.depth = search_depth)
  }

  # Predict on out of sample
  ht1_pred <- predict(ht1, X[-dt,])
  ht2_pred <- predict(ht2, X[dt,])

  g1_util1 <- get_utility(ht1_pred, gamma1[-dt,])[["mean_utility"]]
  g2_util1 <- get_utility(ht1_pred, gamma2[-dt,])[["mean_utility"]]
  g1_util2 <- get_utility(ht2_pred, gamma1[dt,])[["mean_utility"]]
  g2_util2 <- get_utility(ht2_pred, gamma2[dt,])[["mean_utility"]]

  c(mean(g1_util1, g1_util2), mean(g2_util1, g2_util2))
}

  utility_point <- honest_pt(X, gamma1, gamma2, g1_weight, search_depth)
  sd <- map(1:200, function (a) {
          dt = sort(sample.int(nrow(X), nrow(X), replace = TRUE))
          honest_pt(X[dt,], gamma1[dt,], gamma1[dt,], g1_weight, search_depth)
  }
  )

  se1 <- sd(map_dbl(sd, ~.x[1])) / sqrt(200)
  se2 <- sd(map_dbl(sd, ~.x[2])) / sqrt(200)

  # honest_pt(dfX, dfG1, dfG2, 99.0, 1)


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


# honest_pt <- function(X, gamma1, gamma2, g1_weight) {
#   # Add in mix of gammas for fitting
#   gamma = (gamma1 * g1_weight) + ((100.1 - g1_weight) * gamma2)
#   ht = policy_tree(X, gamma, depth = 3)
#
#   png(filename="optimal.png")
#   plot(ht)
#   dev.off()
#   return (0)
# }

# df <- read_csv('train_data1.csv')
# dfX <- df[c('hhh_gender', 'hhh_age', 'hhh_literacy',
#        'age', 'female', 'monthly_spending')]
#
# dfG1 <- df[c('any_drops2-1', 'any_drops3-1', 'any_drops4-1')]  * 0.1446128
#
# dfG2 <- df[c('maths2-1', 'maths3-1', 'maths4-1')] * 0.9120552


# evaluate_tree(dfX, dfG1, dfG2, 99, search_depth = 2)



