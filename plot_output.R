library(tidyverse)
library(plotly)
library(cowplot)

points <- read_csv("pareto_resultsCFG.csv")
points_hybrid <- read_csv("pareto_resultsCFH.csv")
files_paths <- list.files(path="",
                          pattern="^v.*\\.csv$")

points <- map(files_paths, read_csv)
names(points) <- list.files(path="",
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
    a_mean = (a_mean * a_all_sd) + a_all_mean,
    b_mean = (b_mean * b_all_sd) + b_all_mean,
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
  ggplot(aes(x = b_mean, y = a_mean, color = covariate_set)) +
  # geom_point(alpha = 0.1) +
  geom_step(linewidth = 1.3) + 
  labs(y = "Encouraging correct use", x = "Discouraging overuse", color = "Covariate set") +
  facet_grid(rows = vars(depth))

plot4

ggsave("frontiers.png", width = 6, height = 4)


outcomeA <- points %>%
  ggplot(aes(x = parameter, y = a_mean, color = covariate_set)) +
  # geom_point(alpha = 0.1) +
  geom_step(linewidth = 1.3) + 
  labs(y = "Improving access", x = "Weighting hyperparameter", color = "Covariate set") +
  facet_grid(rows = vars(depth))

ggsave("outcomeA.png")


outcomeB <- points %>%
  ggplot(aes(x = parameter, y = b_mean, color = covariate_set)) +
  geom_step(linewidth = 1.3) + 
  labs(y = "Discouraging overuse", x = "Weighting hyperparameter", color = "Covariate set") +
  facet_grid(rows = vars(depth))

hp_value <- plot_grid(outcomeA + theme(legend.position="none"), 
          outcomeB + theme(legend.position="none"),
          get_legend(outcomeA),
          nrow=1,
          rel_widths = c(2, 2, 1.5))

cowplot::save_plot("hp_value.png", hp_value, base_height = 4, base_asp = 2.3)
