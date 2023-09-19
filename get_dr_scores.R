library(tidyverse)
library(grf)
set.seed("20")

data <- read_csv('mopol_data_old.csv')

X <- data %>% select_("hhh_gender", "hhh_age", "monthly_spending", "hhh_literacy", "age", "gender", "benef", "female", "est_num_kids", "f4", "school_spending")
Y <- data %>% select(maths_results_s, any_drops_n)
W <- data$group1

cf <- multi_arm_causal_forest(
  X = X,
  Y = Y,
  W = W %>% as.factor(),
  num.trees = 2000
)

dr_scores <- get_scores(cf) %>%
  as.data.frame()

names(dr_scores) <- c("maths2-1", "maths3-1", "maths4-1", "any_drops2-1", "any_drops3-1", "any_drops4-1")

data <- bind_cols(data, dr_scores)

write_csv(data %>% na.omit(), 'mopol_data.csv')
