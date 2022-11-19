library(tidyverse)
library(policytree)

evaluate_tree <- function(X, gamma1, gamma2, g1_weight, g2_weight) {
  get_utility <- function(predictions, gamma) {
  utilities <- c()
  for (i in 1:nrow(gamma)) utilities <- c(utilities, gamma[i, predictions[i]])
  surrogate_util <- mean(unlist(utilities))
  oracle_util <- gamma %>% apply(1, max) %>% mean()
  return(list(mean_utility = surrogate_util, oracle_utility = oracle_util, regret = oracle_util - surrogate_util))
}

honest_pt <- function(X, gamma1, gamma2, g1_weight, g2_weight) {
  # Add in mix of gammas for fitting
  gamma = (gamma1 * g1_weight) + (gamma2 * g2_weight)
  dt = sort(sample.int(nrow(X), nrow(X)*.7, replace = FALSE))
  ht = hybrid_policy_tree(X[dt,], gamma[dt,], depth = 3)
  ht_pred <- predict(ht, X[-dt,])

  g1_util <- get_utility(ht_pred, gamma1[-dt,])[["mean_utility"]]

  g2_util <- get_utility(ht_pred, gamma2[-dt,])[["mean_utility"]]

  c(g1_util, g2_util)
}

  utility_point <- honest_pt(X, gamma1, gamma2, g1_weight, g2_weight)
  sd <- map(1:50, function (a) {
          dt = sort(sample.int(nrow(X), nrow(X)*.5, replace = TRUE))
          honest_pt(X[dt,], gamma1[dt,], gamma1[dt,], g1_weight, g2_weight)
  }
  )

  se1 <- sd(map_dbl(sd, ~.x[1])) / sqrt(nrow(X))
  se2 <- sd(map_dbl(sd, ~.x[2])) / sqrt(nrow(X))

  return(c(utility_point[1], utility_point[2], se1, se2))
}

#df <- read_csv('train_data.csv')
#dfX <- df[c('hhh_gender', 'hhh_age', 'monthly_spending', 'hhh_literacy',
#       'age', 'gender')]
#dfG <- bind_cols(df[c('...1')], 0.1)
#
#print(evaluate_tree(dfX, dfG, dfG, 0.3, 0.3))
