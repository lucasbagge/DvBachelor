# MODULE: ADVANCED FEATURE ENGINEERING

# GOALS ----
# - LEARN ADVANCED FEATURE ENGINEERING TECHNIQUES & WORKFLOW

# OBJECTIVES ----
# - IMPLEMENT RECIPES PIPELINE
# - APPLY MODELTIME WORKFLOW & VISUALIZE RESULTS
# - COMPARE SPLINE MODEL VS LAG MODEL

# LIBRARIES & DATA ----

# Time Series ML
library(tidymodels)
library(modeltime)

# Core
library(tidyverse)
library(lubridate)
library(timetk)

# Data
data_prepared_tbl <- read_csv2("data/data_final.csv") %>% 
  select(day, spot) %>% 
  mutate(diff_value = c(NA, diff(spot))) %>% 
  mutate(logv = log(1 + diff_value)) %>% 
  mutate(spot_price_dkk_diff = spot-lag(spot)) %>% 
  drop_na()  
  #rename(optin_time=day,  optins_trans=logv)

data_prepared_tbl

# Save Key Params
limit_lower <- 0
limit_upper <- 3650.8
offset      <- 1
std_mean    <- -5.25529020756467
std_sd      <- 1.1109817111334

# 1.0 STEP 1 - CREATE FULL DATA SET ----
# - Extend to Future Window
# - Add any lags to full dataset
# - Add any external regressors to full dataset

horizon    <- 8*7
lag_period <- 8*7
rolling_periods <- c(30, 60, 90)

data_prepared_full_tbl <- subscribers_transformed_tbl %>%
    
    # Add future window
    bind_rows(
        future_frame(.data = ., .date_var = optin_time, .length_out = horizon)
    ) %>%
    
    # Add Autocorrelated Lags
    tk_augment_lags(optins_trans, .lags = lag_period) %>%
    
    # Add rolling features
    tk_augment_slidify(
        .value   = optins_trans_lag56,
        .f       = mean, 
        .period  = rolling_periods,
        .align   = "center",
        .partial = TRUE
    ) %>%
    
    # Add Events
    left_join(learning_labs_prep_tbl, by = c("optin_time" = "event_date")) %>%
    mutate(event = ifelse(is.na(event), 0, event)) %>%
    
    # Format Columns
    rename(lab_event = event) %>%
    rename_with(.cols = contains("lag"), .fn = ~ str_c("lag_", .))


data_prepared_full_tbl %>%
    pivot_longer(-optin_time) %>%
    plot_time_series(optin_time, value, name, .smooth = FALSE)

data_prepared_full_tbl %>% tail(8*7 + 1)

# 2.0 STEP 2 - SEPARATE INTO MODELING & FORECAST DATA ----

data_prepared_full_tbl %>% tail(57)

data_prepared_tbl <- data_prepared_full_tbl %>%
    filter(!is.na(optins_trans))
data_prepared_tbl

forecast_tbl <- data_prepared_full_tbl %>%
    filter(is.na(optins_trans))
forecast_tbl

# 3.0 TRAIN/TEST (MODEL DATASET) ----

data_prepared_tbl

splits <- time_series_split(data_prepared_tbl, assess = horizon, cumulative = TRUE)

splits %>%
    tk_time_series_cv_plan() %>%
    plot_time_series_cv_plan(optin_time, optins_trans)

# 4.0 RECIPES ----
# - Time Series Signature - Adds bulk time-based features
# - Spline Transformation to index.num
# - Interaction: wday.lbl:week2
# - Fourier Features

model_fit_best_lm <- read_rds("00_models/model_fit_best_lm.rds")

model_fit_best_lm %>% summary()

model_fit_best_lm$terms %>% formula()

recipe_spec_base <- recipe(optins_trans ~ ., data = training(splits)) %>%
    
    # Time Series Signature
    step_timeseries_signature(optin_time) %>%
    step_rm(matches("(iso)|(xts)|(hour)|(minute)|(second)|(am.pm)")) %>%
    
    # Standardization
    step_normalize(matches("(index.num)|(year)|(yday)")) %>%
    
    # Dummy Encoding (One Hot Encoding)
    step_dummy(all_nominal(), one_hot = TRUE) %>%
    
    # Interaction
    step_interact(~ matches("week2") * matches("wday.lbl")) %>%
    
    # Fourier
    step_fourier(optin_time, period = c(7, 14, 30, 90, 365), K = 2)


recipe_spec_base %>% prep() %>% juice() %>% glimpse()

# 5.0 SPLINE MODEL ----

# * LM Model Spec ----

model_spec_lm <- linear_reg() %>%
    set_engine("lm")

# OPTIONAL FIX (Overfitting) ----
# - LINEAR REGRESSION WITH MANY PARAMETERS CAN OVERFIT
# - SOLUTION: WE CAN SWITCH TO PENALIZED REGRESSION WITH ELASTIC NET
model_spec_lm <- linear_reg(penalty = 0.01) %>%
    set_engine("glmnet")
# END OPTIONAL FIX ----

# * Spline Recipe Spec ----

recipe_spec_base %>% prep() %>% juice() %>% glimpse()

recipe_spec_1 <- recipe_spec_base %>%
    step_rm(optin_time) %>%
    step_ns(ends_with("index.num"), deg_free = 2) %>%
    step_rm(starts_with("lag_"))

recipe_spec_1 %>% prep() %>% juice() %>% glimpse()

# * Spline Workflow  ----

workflow_fit_lm_1_spline <- workflow() %>%
    add_model(model_spec_lm) %>%
    add_recipe(recipe_spec_1) %>%
    fit(training(splits))

workflow_fit_lm_1_spline

workflow_fit_lm_1_spline %>% 
    pull_workflow_fit() %>%
    pluck("fit") %>%
    summary()

# 6.0 MODELTIME  ----

calibration_tbl <- modeltime_table(
    workflow_fit_lm_1_spline
) %>%
    modeltime_calibrate(new_data = testing(splits))

calibration_tbl %>%
    modeltime_forecast(new_data    = testing(splits), 
                       actual_data = data_prepared_tbl) %>%
    plot_modeltime_forecast()

calibration_tbl %>% modeltime_accuracy()

# 7.0 LAG MODEL ----

# * Lag Recipe ----

recipe_spec_base %>% prep() %>% juice() %>% glimpse()

recipe_spec_2 <- recipe_spec_base %>%
    step_rm(optin_time) %>%
    step_naomit(starts_with("lag_"))
    

recipe_spec_2 %>% prep() %>% juice() %>% glimpse()

# * Lag Workflow ----

workflow_fit_lm_2_lag <- workflow() %>%
    add_model(model_spec_lm) %>%
    add_recipe(recipe_spec_2) %>%
    fit(training(splits))

workflow_fit_lm_2_lag

workflow_fit_lm_2_lag %>% pull_workflow_fit() %>% pluck("fit") %>% summary()

# * Compare with Modeltime -----

calibration_tbl <- modeltime_table(
    workflow_fit_lm_1_spline,
    workflow_fit_lm_2_lag
) %>%
    modeltime_calibrate(new_data = testing(splits))

calibration_tbl %>%
    modeltime_forecast(new_data    = testing(splits), 
                       actual_data = data_prepared_tbl) %>%
    plot_modeltime_forecast()

calibration_tbl %>%
    modeltime_accuracy()

# 8.0 FUTURE FORECAST ----

refit_tbl <- calibration_tbl %>%
    modeltime_refit(data = data_prepared_tbl)

refit_tbl %>%
    modeltime_forecast(new_data    = forecast_tbl,
                       actual_data = data_prepared_tbl) %>%
    
    # Invert Transformation
    mutate(across(.value:.conf_hi, .fns = ~ standardize_inv_vec(
        x    = .,
        mean = std_mean,
        sd   = std_sd
    ))) %>%
    mutate(across(.value:.conf_hi, .fns = ~ log_interval_inv_vec(
        x           = ., 
        limit_lower = limit_lower, 
        limit_upper = limit_upper, 
        offset      = offset
    ))) %>%
    
    plot_modeltime_forecast()

# 9.0 SAVE ARTIFACTS ----


feature_engineering_artifacts_list <- list(
    # Data
    data = list(
        data_prepared_tbl = data_prepared_tbl,
        forecast_tbl      = forecast_tbl 
    ),
    
    # Recipes
    recipes = list(
        recipe_spec_base = recipe_spec_base,
        recipe_spec_1    = recipe_spec_1, 
        recipe_spec_2    = recipe_spec_2
    ),
    
    # Models / Workflows
    models = list(
        workflow_fit_lm_1_spline = workflow_fit_lm_1_spline,
        workflow_fit_lm_2_lag    = workflow_fit_lm_2_lag
    ),
    
    
    # Inversion Parameters
    standardize = list(
        std_mean = std_mean,
        std_sd   = std_sd
    ),
    log_interval = list(
        limit_lower = limit_lower, 
        limit_upper = limit_upper,
        offset      = offset
    )
    
)

feature_engineering_artifacts_list

feature_engineering_artifacts_list %>% 
    write_rds("00_models/feature_engineering_artifacts_list.rds")

read_rds("00_models/feature_engineering_artifacts_list.rds")
