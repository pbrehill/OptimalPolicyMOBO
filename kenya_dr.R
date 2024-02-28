library(dplyr)
library(haven)
library(grf)
library(readr)
library(policytree)



impute_median <- function(df) {
  df %>%
    mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))
}


set.seed(99)

# This file creates the doubly robust scores for use in policy learning

# Data available from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EQJPZT

ken_main <- "ACT_AllMain_FINAL_pub.dta" %>% read_dta() %>%
  filter(coartemprice != 500)


Y <- ken_main %>%
  transmute(
    yes_treated = ifelse(used_act == 1 & LOG_mal_prob21 >= 0.7, 1, 0),
    no_treated = ifelse(used_act == 1 & LOG_mal_prob21 < 0.3, -1, 0)
  ) %>%
  mutate_all(scale)

row_selector <- Y %>% complete.cases()

X <- ken_main %>%
  select(head_fem:B_mal_adult_hh, LOG_patient_age1)

W <- ken_main$coartemprice %>%
  as.factor()

ken_forest <- multi_arm_causal_forest(
  X = X[row_selector,], Y = Y[row_selector,], W = W[row_selector], num.trees = 20000
)

gamma1 <- double_robust_scores(ken_forest, 1) %>% as.data.frame.matrix()
names(gamma1) <- paste0(names(gamma1), ".yes_treated")

gamma2 <- double_robust_scores(ken_forest, 2)  %>% as.data.frame.matrix()
names(gamma2) <- paste0(names(gamma2), ".no_treated")

gamma <- bind_cols(gamma1, gamma2)
write_csv(gamma, "kenya_dr.csv")
write_csv(X[row_selector,] %>% 
            impute_median(), "kenya_covars.csv")
write_csv(X[row_selector,] %>% 
            impute_median() %>%
            select(B_head_age_imputed, dist_clinic, head_acres, head_mar, head_dep, subfarm, B_hh_size, B_adultteen, LOG_patient_age1), 
          "kenya_limited_covars.csv")



