library(readr)
library(purrr)
library(dplyr)
library(stringr)
library(ggplot2)
library(cowplot)
library(tidyr)


# points <- read_csv("pareto_resultsCFG.csv")
# points_hybrid <- read_csv("pareto_resultsCFH.csv")
files_paths <- list.files(
                          pattern="^v.*\\.csv$")

points <- map(files_paths, read_csv)
names(points) <- list.files(
                               pattern="^v.*\\.csv$")

# Get original data to un-standardise
Y <- ken_main %>%
  transmute(
    yes_treated = ifelse(used_act == 1 & LOG_mal_prob21 >= 0.5, 1, 0),
    no_treated = ifelse(used_act == 1 & LOG_mal_prob21 < 0.5, -1, 0)
  )

a_all_mean <- mean(Y$yes_treated, na.rm = T)
b_all_mean <- mean(Y$no_treated, na.rm = T)

a_all_sd <- sd(Y$yes_treated, na.rm = T)
b_all_sd <- sd(Y$no_treated, na.rm = T)

# Make figures

points <- points %>%
  bind_rows(.id = "trial") %>%
  mutate(
    # Unstandardising outcomes and standard errors
    a_mean = (a_mean * a_all_sd) + a_all_mean,
    b_mean = (b_mean * b_all_sd) + b_all_mean,
    a_sem = (a_sem * a_all_sd),
    b_sem = (b_sem * b_all_sd),
    depth = paste0("Depth ", extract_numeric(trial)),
    covariate_set = ifelse(str_detect(trial, "vb"),
                           "Full covariates",
                           ifelse(
                             str_detect(trial, "vs"),
                             "Medium covariates",
                             "Age and distance only"
                           )
                           ) %>% as.factor()
  )

opts <- read_csv("opt_evaluations.csv") %>%
  mutate(depth = 3)

plot4 <- points %>%
  ggplot(aes(x = b_mean, y = a_mean, group = covariate_set)) +
  # geom_point(alpha = 0.1) +
  geom_line(aes(y = a_mean, color = covariate_set, linetype = covariate_set)) + 
  labs(y = "Discouraging overuse", 
       x = "Encouraging correct use", 
       color = "Covariate set",
       linetype = "Covariate set"
  ) +
  facet_grid(rows = vars(depth))

plot4

ggsave("frontiers.png", width = 6, height = 4)


outcomeA <- points %>%
  ggplot(aes(x = parameter, y = a_mean)) +
  # geom_point(alpha = 0.1) +
  geom_ribbon(aes(ymin = a_mean - a_sem, ymax = a_mean + a_sem, fill = covariate_set), alpha = 0.5) +
  geom_line(aes(color = covariate_set, linetype = covariate_set), linewidth = 1) + 
  labs(y = "Discouraging overuse", 
       x = "Weighting hyperparameter", 
       color = "Covariate set",
       linetype = "Covariate set",
       fill = "Covariate set"
  ) +
  facet_grid(rows = vars(depth))

ggsave("outcomeA.png")


outcomeB <- points %>%
  ggplot(aes(x = parameter, y = b_mean)) +
  geom_ribbon(aes(ymin = b_mean - b_sem, ymax = b_mean + b_sem, fill = covariate_set), alpha = 0.5) +
  geom_line(aes(color = covariate_set, linetype = covariate_set), linewidth = 1) + 
  labs(y = "Discouraging overuse", 
       x = "Weighting hyperparameter", 
       color = "Covariate set",
       linetype = "Covariate set",
       fill = "Covariate set"
       ) +
  facet_grid(rows = vars(depth))

hp_value <- plot_grid(outcomeA + theme(legend.position="none"),
          outcomeB + theme(legend.position="none"),
          get_legend(outcomeA),
          nrow=1,
          rel_widths = c(2, 2, 1.5))

cowplot::save_plot("hp_value.png", hp_value, base_height = 4, base_asp = 2.3)

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

