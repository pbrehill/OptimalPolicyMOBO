source('tree_utility.R')

set.seed(21)

#   get_utility <- function(predictions, gamma) {
#   utilities <- c()
#   for (i in 1:nrow(gamma)) utilities <- c(utilities, gamma[i, predictions[i]])
#   surrogate_util <- mean(unlist(utilities))
#   oracle_util <- gamma %>% apply(1, max) %>% mean()
#   return(list(mean_utility = surrogate_util, oracle_utility = oracle_util, regret = oracle_util - surrogate_util))
# }
#
# honest_opt <- function(X, gamma1, gamma2, g1_weight) {
#   # Add in mix of gammas for fitting
#   gamma = (gamma1 * g1_weight) + ((100.1 - g1_weight) * gamma2)
#   dt = sort(sample.int(nrow(X), nrow(X)*.7, replace = FALSE))
#   ht = policy_tree(X[dt,], gamma[dt,], depth = 3)
#
#   # Predict on out of sample
#   ht_pred <- predict(ht, X[-dt,])
#
#   g1_util <- get_utility(ht_pred, gamma1[-dt,])[["mean_utility"]]
#
#   g2_util <- get_utility(ht_pred, gamma2[-dt,])[["mean_utility"]]
#
#   list(ht, c(g1_util, g2_util))
# }
#
# honest_hpt <- function(X, gamma1, gamma2, g1_weight, search_depth) {
#   # Add in mix of gammas for fitting
#   gamma = (gamma1 * g1_weight) + ((100.1 - g1_weight) * gamma2)
#   dt = sort(sample.int(nrow(X), nrow(X)*.7, replace = FALSE))
#   ht = hybrid_policy_tree(X[dt,], gamma[dt,], depth = 3, search.depth = search_depth)
#   plot(ht)
#
#   # Predict on out of sample
#   ht_pred <- predict(ht, X[-dt,])
#
#   g1_util <- get_utility(ht_pred, gamma1[-dt,])[["mean_utility"]]
#
#   g2_util <- get_utility(ht_pred, gamma2[-dt,])[["mean_utility"]]
#
#   c(g1_util, g2_util)
# }

  # utility_point <- honest_pt(X, gamma1, gamma2, g1_weight, search_depth)
  # sd <- map(1:100, function (a) {
  #         dt = sort(sample.int(nrow(X), nrow(X)*.5, replace = TRUE))
  #         honest_pt(X[dt,], gamma1[dt,], gamma1[dt,], 96.19237563, search_depth)
  # }
  # )
  #
  # se1 <- sd(map_dbl(sd, ~.x[1])) / sqrt(nrow(X))
  # se2 <- sd(map_dbl(sd, ~.x[2])) / sqrt(nrow(X))


frontier_points <- read_csv('all_pareto_results.csv')

df <- read_csv('train_data1.csv')
dfX <- df[c('hhh_gender', 'hhh_age', 'hhh_literacy',
       'age', 'female', 'monthly_spending')]

dfG1 <- df[c('any_drops2-1', 'any_drops3-1', 'any_drops4-1')]  * 0.1446128

dfG2 <- df[c('maths2-1', 'maths3-1', 'maths4-1')] * 0.9120552

sampled <- frontier_points %>%
  filter(Surrogate == "Hybrid") %>%
  select(Parameter) %>%
  pull() %>%
  sample(10)

# start_time <- Sys.time()
# res <- map(sampled, ~honest_opt(dfX, dfG1, dfG2, .x))
# end_time <- Sys.time()
# print(end_time - start_time)
#
# write_csv(data.frame(
#           Parameter = sampled,
#           a = map_dbl(res, ~.x[1]),
#           b = map_dbl(res, ~.x[2])
# ), 'opt_out.csv')


get_oracles <- function(gamma1, gamma2, g1_weight) {
  # Get the weighted utility matrix
  gamma <- (gamma1 * g1_weight) + ((100.1 - g1_weight) * gamma2)

  # Find the maximum
  maxes <- max.col(gamma)

  # Calculate seperate utilities for the treatment
  g1_total <- c()
  for (i in 1:length(maxes)) g1_total <- c(g1_total , gamma1[i, maxes[i]])

  g2_total <- c()
  for (i in 1:length(maxes)) g2_total <- c(g2_total , gamma2[i, maxes[i]])

  return(list(g1 = g1_total, g2 = g2_total))

}

# params <- frontier_points %>%
#   select(Parameter) %>%
#   pull()
#
# oracs <- map(frontier_points %>%
#   select(Parameter) %>%
#   pull(), ~get_oracles(dfG1, dfG2, .x))

utilities_as_weighted <- function (g_bar, g1_weight) {
  (g_bar[1] * g1_weight) + ((100.1 - g1_weight) * g_bar[1] )
}



# write_csv(data.frame(
#           Parameter = params,
#           orig_surrogate = frontier_points %>%
#   select(Surrogate) %>%
#   pull(),
#           a = map_dbl(oracs, ~.x[[1]] %>% mean()),
#           b = map_dbl(oracs, ~.x[[2]] %>% mean())
# ), 'orac_out.csv')

# start_time <- Sys.time()
# res <- policy_tree(dfX, (dfG1 * 97.7406224
# ) + ((100.1 - 97.7406224
# ) * dfG2), depth = 3)
# end_time <- Sys.time()
# print(end_time - start_time)

df <- read_csv('train_data1.csv')
dfX <- df[c('hhh_gender', 'hhh_age', 'hhh_literacy',
       'age', 'female', 'monthly_spending')]

dfG1 <- df[c('any_drops2-1', 'any_drops3-1', 'any_drops4-1')]  * 0.1446128

dfG2 <- df[c('maths2-1', 'maths3-1', 'maths4-1')] * 0.9120552

params <- read_csv('params.csv')
outputs <- map(params$Parameter, ~get_oracles(dfG1, dfG2, .x))
oracle_out <- data.frame(param = params$Parameter,
           g1 = map_dbl(outputs, ~.x[1]$g1 %>% mean()),
           g2 = map_dbl(outputs, ~.x[2]$g2 %>% mean())
)
print('stop')
