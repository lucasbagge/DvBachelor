# MODULE: ARIMA MODELING

# GOAL: Understand ARIMA

# OBJECTIVES ----
# - Describe ARIMA using Linear Regression


# LIBRARIES & SETUP ----

# Time Series ML
library(tidymodels)
library(modeltime)

# Core 
library(tidyverse)
library(lubridate)
library(timetk)

# DATA ----

data_prepared_tbl <- read_csv2("data/data_final.csv") %>% 
  select(day, spot) %>% 
  mutate(diff_value = c(NA, diff(spot))) %>% 
  mutate(logv = log(1 + diff_value)) %>% 
  drop_na() %>% 
  rename(optin_time=day,  optins_trans=logv)

# TRAIN / TEST SPLITS ----

splits <- time_series_split(data_prepared_tbl, assess = "90 weeks", cumulative = TRUE)

splits %>%
    tk_time_series_cv_plan() %>%
    plot_time_series_cv_plan(optin_time, optins_trans)

train_tbl <- training(splits) %>%
    select(optin_time, optins_trans)


# 1.0 CONCEPTS ----


model_fit_arima <- arima_reg(
    seasonal_period = 7,
    non_seasonal_ar = 1,
    non_seasonal_differences = 1,
    non_seasonal_ma = 1,
    seasonal_ar = 1,
    seasonal_differences = 1,
    seasonal_ma = 1) %>%
    set_engine("arima") %>%
    fit(optins_trans ~ optin_time, training(splits))

modeltime_table(
    model_fit_arima) %>%
    modeltime_calibrate(testing(splits)) %>%
    modeltime_forecast(
        new_data = testing(splits),
        actual_data = data_prepared_tbl) %>%
    plot_modeltime_forecast()

# 3.0 AUTO ARIMA + XREGS ----

# * Model ----

model_fit_auto_arima <- arima_reg() %>%
    set_engine("auto_arima") %>%
    fit(
        optins_trans ~ optin_time 
        + fourier_vec(optin_time, period = 7)
        + fourier_vec(optin_time, period = 14)
        + fourier_vec(optin_time, period = 30)
        + fourier_vec(optin_time, period = 90)
        + month(optin_time, label = TRUE),
        data = training(splits)
    )

# * Calibrate ----

calibration_tbl <- modeltime_table(
    model_fit_arima,
    model_fit_auto_arima) %>%
    modeltime_calibrate(testing(splits))

# * Forecast Test ----

calibration_tbl %>%
    modeltime_forecast(
        new_data = testing(splits),
        actual_data = data_prepared_tbl
    ) %>%
    plot_modeltime_forecast()

# * Accuracy Test -----
calibration_tbl %>% modeltime_accuracy()

