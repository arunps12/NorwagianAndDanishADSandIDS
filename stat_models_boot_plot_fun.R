
# Load necessary libraries Libraries
library(readxl)
library(lme4)
library(lmerTest)
library(ggplot2)
library(dplyr)
library(crayon)
library(merTools)
library(boot)
library(splines)
library(stringr)
library(glmmTMB)
library(car)
library(DHARMa)
library(performance)
library(emmeans)
# Functions for stat models, boot, and plotting
# 1. lmer model
lmer.full.reduced.null.compare <- function(fullmformula, 
                                           optimizer = "bobyqa", 
                                           maxfun = 100000,
                                           fullm = TRUE,
                                           redm = FALSE, 
                                           redmformula = NULL, 
                                           nullm = FALSE, 
                                           nullmformula = NULL,
                                           data_list = list(...),
                                           marginal_means = FALSE,
                                           specs = NULL,
                                           ...) {
  
  # Load required packages
  if (!requireNamespace("lme4", quietly = TRUE)) {
    stop("The 'lme4' package is required but is not installed.")
  }
  if (!requireNamespace("lmerTest", quietly = TRUE)) {
    stop("The 'lmerTest' package is required but is not installed.")
  }
  if (!requireNamespace("car", quietly = TRUE)) {
    stop("The 'car' package is required but is not installed.")
  }
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("The 'ggplot2' package is required but is not installed.")
  }
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("The 'dplyr' package is required but is not installed.")
  }
  if (!requireNamespace("emmeans", quietly = TRUE)) {
    stop("The 'emmeans' package is required but is not installed.")
  }
  if (!requireNamespace("crayon", quietly = TRUE)) {
    stop("The 'crayon' package is required but is not installed.")
  }
  
  library(lme4)
  library(lmerTest)
  library(car)
  library(ggplot2)
  library(dplyr)
  library(emmeans)
  library(crayon)
  
  # Combine input dataframes
  combined_data <- bind_rows(data_list)
  cat(blue("Structure of combined data:\n"))
  str(combined_data)
  
  # Extract the response variable 
  terms_obj <- terms(fullmformula)
  response_var <- as.character(attr(terms_obj, "variables"))[2]
  cat(blue("Response variable is:"), response_var, "\n")
  
  # Standardize all numeric columns except the response variable
  numeric_columns <- sapply(combined_data, is.numeric)
  numeric_columns <- setdiff(names(combined_data)[which(numeric_columns)], response_var)
  combined_data[numeric_columns] <- lapply(combined_data[numeric_columns], scale)
  
  # Check structure of combined data after scaling
  cat(blue("Structure of combined data after scaling:\n"))
  str(combined_data)
  
  # Histogram of Response variable
  p_histogram <- ggplot(combined_data, aes_string(x = response_var)) +
    geom_histogram(bins = 30) +
    labs(title = "Histogram of Response Variable")
  print(p_histogram)  
  
  # Check if the response variable is normally distributed
  if (shapiro.test(combined_data[[response_var]])$p.value < 0.05) {
    combined_data[[response_var]] <- log(combined_data[[response_var]] + 1)
    cat(blue("Response variable is not normally distributed. Log transformation applied.\n"))
    cat(blue("Structure of combined data after transformation:\n"))
    str(combined_data)
    
    # Histogram of Response variable after transformation
    p_histogram <- ggplot(combined_data, aes_string(x = response_var)) +
      geom_histogram(bins = 30) +
      labs(title = "Histogram of Log-transformed Response Variable")
    print(p_histogram)  
  } else {
    cat(blue("Response variable is normally distributed; no transformation applied.\n"))
  }
  
  # Initialize list to store emmeans
  #emmeans_list <- list()
  
  # Fitting the full model
  fit_model <- NULL
  if (fullm && !is.null(fullmformula)) {
    cat(blue("Fitting Full Model.\n"))
    fit_model <- tryCatch({
      lmer(fullmformula, data = combined_data, REML = FALSE, 
           control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
    }, error = function(e) {
      cat(blue("Error in fitting the full model:", e$message, "\n"))
      return(NULL)
    })
    
    # Check if the full model was fitted successfully
    if (is.null(fit_model)) {
      cat(blue("Full model failed. Consider using a different optimizer or increasing maxfun.\n"))
    } else {
      fit_model_REML <- lmerTest::lmer(fullmformula, data = combined_data, REML = TRUE, 
                                       control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
      
      # Check for singular fit
      if (isSingular(fit_model)) {
        cat(blue("Full model is singular. Consider simplifying the random effects structure.\n"))
      }
      
      # Residuals and fitted values plots for full model
      residuals_full <- resid(fit_model)
      fitted_values_full <- fitted(fit_model)
      
      p_residuals_vs_fitted <- ggplot(data = data.frame(fitted = fitted_values_full, residuals = residuals_full), aes(x = fitted, y = residuals)) +
        geom_point() +
        geom_smooth(method = "lm") +
        labs(title = "Residuals vs Fitted Values for Full Model", x = "Fitted Values", y = "Residuals")
      print(p_residuals_vs_fitted)  
      
      p_qq_plot <- ggplot(data = data.frame(residuals = residuals_full), aes(sample = residuals)) +
        geom_qq() +
        geom_qq_line() +
        labs(title = "Q-Q Plot of Residuals for Full Model")
      print(p_qq_plot)  
      
      # Extract model summary for full model
      model_summary <- summary(fit_model_REML)
      cat(blue("Summary of the Full Model:\n"))
      print(model_summary)
      
      logLik_full <- logLik(fit_model)
      df_full <- attr(logLik_full, "df")
      cat(blue("Full Model: Log-Likelihood =", round(as.numeric(logLik_full), 2), 
               ", Degrees of Freedom =", df_full, "\n"))
      vif_full <- vif(fit_model)
      cat(blue("Max VIF value of the Full Model:", max(vif_full), "\n"))
      
      # Compute emmeans for full model if requested
      if (marginal_means) {
        if (is.null(specs)) {
          stop("Please provide 'specs' argument for computing emmeans.")
        }
        cat(blue("Computing estimated marginal means for the Full Model.\n"))
        emm <- emmeans(fit_model, specs = specs)
        print(summary(emm))
        #emmeans_list$full_model_emm <- emm_full
      }
    }
  }
  
  # Fitting the reduced model
  red_fit_model <- NULL
  if (redm && !is.null(redmformula)) {
    cat(blue("Fitting Reduced Model.\n"))
    red_fit_model <- tryCatch({
      lmer(redmformula, data = combined_data, REML = FALSE, 
           control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
    }, error = function(e) {
      cat(blue("Error in fitting the reduced model:", e$message, "\n"))
      return(NULL)
    })
    
    if (is.null(red_fit_model)) {
      cat(blue("Reduced model failed. Consider using a different optimizer or increasing maxfun.\n"))
    } else {
      red_fit_model_REML <- lmerTest::lmer(redmformula, data = combined_data, REML = TRUE, 
                                           control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
      
      # Check for singular fit
      if (isSingular(red_fit_model)) {
        cat(blue("Reduced model is singular. Consider simplifying the random effects structure.\n"))
      }
      
      # Residuals and fitted values plots for reduced model
      residuals_red <- resid(red_fit_model)
      fitted_values_red <- fitted(red_fit_model)
      
      p_residuals_vs_fitted <- ggplot(data = data.frame(fitted = fitted_values_red, residuals = residuals_red), aes(x = fitted, y = residuals)) +
        geom_point() +
        geom_smooth(method = "lm") +
        labs(title = "Residuals vs Fitted Values for Reduced Model", x = "Fitted Values", y = "Residuals")
      print(p_residuals_vs_fitted)  
      
      p_qq_plot <- ggplot(data = data.frame(residuals = residuals_red), aes(sample = residuals)) +
        geom_qq() +
        geom_qq_line() +
        labs(title = "Q-Q Plot of Residuals for Reduced Model")
      print(p_qq_plot)  
      
      # Extract model summary for reduced model
      red_model_summary <- summary(red_fit_model_REML)
      cat(blue("Summary of the Reduced Model:\n"))
      print(red_model_summary)
      
      logLik_reduced <- logLik(red_fit_model)
      df_reduced <- attr(logLik_reduced, "df") 
      cat(blue("Reduced Model: Log-Likelihood =", round(as.numeric(logLik_reduced), 2), 
               ", Degrees of Freedom =", df_reduced, "\n"))
      vif_red <- vif(red_fit_model)
      cat(blue("Max VIF value of the Reduced Model:", max(vif_red), "\n"))
      
      # Compute emmeans for reduced model if requested
      if (marginal_means) {
        if (is.null(specs)) {
          stop("Please provide 'specs' argument for computing emmeans.")
        }
        cat(blue("Computing estimated marginal means for the Reduced Model.\n"))
        emm <- emmeans(red_fit_model, specs = specs)
        print(summary(emm))
        #emmeans_list$reduced_model_emm <- emm_red
      }
    }
  }
  
  # Fitting the null model
  null_fit_model <- NULL
  if (nullm && !is.null(nullmformula)) {
    cat(blue("Fitting Null Model.\n"))
    null_fit_model <- tryCatch({
      lmer(nullmformula, data = combined_data, REML = FALSE, 
           control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
    }, error = function(e) {
      cat(blue("Error in fitting null model:", e$message, "\n"))
      return(NULL)
    })
  }
  
  # Compare models using ANOVA
  if (nullm && !is.null(nullmformula)) {
    if (redm && !is.null(redmformula) && !is.null(red_fit_model) && !is.null(null_fit_model)) {
      cat(blue("Comparing Reduced and Null Models using ANOVA (Likelihood Ratio Test):\n"))
      reduced_null_model_comparison <- anova(red_fit_model, null_fit_model)
      print(reduced_null_model_comparison)
    } else if (fullm && !is.null(fullmformula) && !is.null(fit_model) && !is.null(null_fit_model)) {
      cat(blue("Comparing Full and Null Models using ANOVA (Likelihood Ratio Test):\n"))
      full_null_model_comparison <- anova(fit_model, null_fit_model)
      print(full_null_model_comparison)
    }
  }
  
  # Return the results
  cat(blue("Returning the fitted models, data, optimizer settings, and emmeans (if computed).\n"))
  return(list(full_model = fit_model, 
              reduced_model = red_fit_model, 
              null_model = null_fit_model, 
              fit_data = combined_data, 
              optimizer = optimizer, 
              maxfun = maxfun, 
              emmeans = emm))
}





# glmmTB model
glmmTB.full.reduced.null.compare <- function(fullmformula,
                                             family,
                                             zi_formula = NULL,
                                             fullm = FALSE,
                                             redm = FALSE, 
                                             redmformula = NULL, 
                                             nullm = FALSE, 
                                             nullmformula = NULL,
                                             data_list = list(...),
                                             marginal_means = FALSE,
                                             specs = NULL,
                                             ...) {
  # Load required packages
  if (!requireNamespace("glmmTMB", quietly = TRUE)) {
    stop("The 'glmmTMB' package is required but is not installed.")
  }
  if (!requireNamespace("emmeans", quietly = TRUE)) {
    stop("The 'emmeans' package is required but is not installed.")
  }
  if (!requireNamespace("crayon", quietly = TRUE)) {
    stop("The 'crayon' package is required but is not installed.")
  }
  
  library(glmmTMB)
  library(emmeans)
  library(crayon)
  
  # Combine input dataframes
  combined_data <- bind_rows(data_list)
  cat(blue("Structure of combined data:\n"))
  str(combined_data)
  
  # Extract the response variable 
  terms_obj <- terms(fullmformula)
  response_var <- as.character(attr(terms_obj, "variables"))[2]  # Extract the response variable
  print(response_var)
  
  # Standardize all numeric columns except the response variable
  numeric_columns <- sapply(combined_data, is.numeric)
  numeric_columns <- setdiff(names(combined_data)[which(numeric_columns)], response_var)
  combined_data[numeric_columns] <- lapply(combined_data[numeric_columns], scale)  # Standardize numeric columns
  
  # Check structure of combined data after scaling
  cat(blue("Structure of combined data after scaling:\n"))
  str(combined_data)
  
  # Histogram of Response variable
  p_histogram <- ggplot(combined_data, aes_string(x = response_var)) +
    geom_histogram(bins = 30) +
    labs(title = "Histogram of Response Variable")
  print(p_histogram)
  
  # Initialize emmeans variable to NULL
  #emm <- NULL
  # Fit the full model using glmmTMB
  fit_model <- NULL
  if (fullm && !is.null(fullmformula)) {
    cat(blue("Fitting Full Model.\n"))
    fit_model <- tryCatch({
      glmmTMB(formula = fullmformula, data = combined_data, family = family, ziformula = zi_formula) 
    }, error = function(e) {
      cat(blue("Error in fitting full model:", e$message, "\n"))
      return(NULL)
    })
    full_model_summary <- summary(fit_model)
    cat(blue("Summary of the full model:\n"))
    print(full_model_summary)
    cat(blue("Simulate residuals for the fitted full glmmTMB model.\n"))
    simulated_residuals_full <- simulateResiduals(fittedModel = fit_model, plot = T)
    
    cat(blue("Check for overdispersion with DHARMa test for full model.\n"))
    dispersion_test_full <- DHARMa::testDispersion(simulationOutput = simulated_residuals_full)
    
    # Print the results of the dispersion test
    cat("DHARMa dispersion for full model test results:\n")
    cat("Dispersion:", dispersion_test_full$statistic, "\n")
    cat("p-value:", dispersion_test_full$p.value, "\n")
    cat("Alternative hypothesis:", dispersion_test_full$alternative, "\n")
    
    # Compute VIF (multicollinearity check)
    collinearity_results_full <- check_collinearity(fit_model)
    cat(blue("VIF for the full model:\n"))
    print(collinearity_results_full)
    
    logLik_full <- logLik(fit_model)
    df_full <- attr(logLik_full, "df")
    cat(blue("Full Model: Log-Likelihood =", round(as.numeric(logLik_full), 2), ", Degrees of Freedom =", df_full, "\n"))
    
    # Compute marginal means if requested
    if (marginal_means) {
      if (is.null(specs)) {
        stop("Please provide 'specs' argument for computing emmeans.")
      }
      cat(blue("Computing estimated marginal means for the Full Model.\n"))
      emm <- emmeans(fit_model, specs = specs, type = "response")
      print(summary(emm))
    }
  }
  
  #if (fullm && !is.null(fullmformula)) {
    
  #}
  
  # Fit the reduced model if specified
  red_fit_model <- NULL
  if (redm && !is.null(redmformula)) {
    cat(blue("Fitting Reduced Model.\n"))
    red_fit_model <- tryCatch({
      glmmTMB(formula = redmformula, data = combined_data, family = family, ziformula = zi_formula)
    }, error = function(e) {
      cat(blue("Error in fitting reduced model:", e$message, "\n"))
      return(NULL)
    })
    red_model_summary <- summary(red_fit_model)
    cat(blue("Summary of the reduced model:\n"))
    print(red_model_summary)
    cat(blue("Simulate residuals for the fitted reduced glmmTMB model.\n"))
    simulated_residuals_red <- simulateResiduals(fittedModel = red_fit_model, plot = T)
    
    cat(blue("Check for overdispersion with DHARMa test for reduced model.\n"))
    dispersion_test_red <- DHARMa::testDispersion(simulationOutput = simulated_residuals_red)
    
    # Print the results of the dispersion test
    cat("DHARMa dispersion for reduced model test results:\n")
    cat("Dispersion:", dispersion_test_red$statistic, "\n")
    cat("p-value:", dispersion_test_red$p.value, "\n")
    cat("Alternative hypothesis:", dispersion_test_red$alternative, "\n")
    
    # Compute VIF (multicollinearity check)
    collinearity_results_red <- check_collinearity(red_fit_model)
    cat(blue("VIF for the reduced model:\n"))
    print(collinearity_results_red)
    
    logLik_red <- logLik(red_fit_model)
    df_red <- attr(logLik_red, "df")
    cat(blue("Reduced Model: Log-Likelihood =", round(as.numeric(logLik_red), 2), ", Degrees of Freedom =", df_red, "\n"))
    
    # Compute marginal means if requested
    if (marginal_means) {
      if (is.null(specs)) {
        stop("Please provide 'specs' argument for computing emmeans.")
      }
      cat(blue("Computing estimated marginal means for the Reduced Model.\n"))
      emm <- emmeans(red_fit_model, specs = specs, type = "response")
      print(summary(emm))
    }
  }
  
  #if (redm && !is.null(redmformula)) {
    
  #}
  
  # Fit the null model if specified
  null_fit_model <- NULL
  if (nullm && !is.null(nullmformula)) {
    cat(blue("Fitting Null Model.\n"))
    null_fit_model <- tryCatch({
      glmmTMB(formula = nullmformula, data = combined_data, family = family, ziformula = zi_formula)
    }, error = function(e) {
      cat(blue("Error in fitting null model:", e$message, "\n"))
      return(NULL)
    })
  }
  
  # Compare models using ANOVA
  if (nullm && !is.null(nullmformula)) {
    if (redm && !is.null(redmformula) && !is.null(red_fit_model) && !is.null(null_fit_model)) {
      cat(blue("Comparing Reduced and Null Models using ANOVA (Likelihood Ratio Test):\n"))
      reduced_null_model_comparison <- anova(red_fit_model, null_fit_model)
      print(reduced_null_model_comparison)
    } else if (fullm && !is.null(fullmformula) && !is.null(fit_model) && !is.null(null_fit_model)) {
      cat(blue("Comparing Full and Null Models using ANOVA (Likelihood Ratio Test):\n"))
      full_null_model_comparison <- anova(fit_model, null_fit_model)
      print(full_null_model_comparison)
    }
  }
  
  # Return the fitted models
  return(list(full_model = fit_model, reduced_model = red_fit_model, null_model = null_fit_model, fit_data = combined_data, emmeans = emm))
}


# 3. fun for bootstrapped predictions

boot.ci.predict.lmer <- function(m, optimizer, maxfun, data, pred.data, reqcol, centercol = NULL, nboots = NULL, link = "identity", keep.boots = FALSE) {
  
  
  # Initialize a counter singular fit
  #warning_counter <- 0
  singular_fit_counter <- 0
  
  # Filter pred.data to use only variables specified in 'reqcol'
  pred.data <- pred.data[ , reqcol, drop = FALSE]
  #print(pred.data)
  
  # Centering variables
  if (!is.null(centercol)) {
    for (var in centercol) {
      if (!is.numeric(data[[var]])) {
        data[[var]] <- as.numeric(as.factor(data[[var]]))  # Convert factor to numeric
      }
    }
    
    # Center the data
    centered_data <- scale(data[centercol], center = TRUE, scale = FALSE)
    means <- attr(centered_data, "scaled:center")  # Extract the mean values
    for (var in centercol) {
      pred.data[[var]] <- rep(means[var], nrow(pred.data))  # Repeat the mean value for each row
    }
  }
  
  # Predict values based on the model
  prediction <- predict(m, newdata = pred.data, type = "response", re.form = ~0)
  #print(prediction)
  
  # Perform bootstrapping to calculate confidence intervals
  if (!is.null(nboots)) {
    set.seed(123)
    boots_results <- boot(data, statistic = function(data, indices) {
      boot_model <- withCallingHandlers(
        {
          update(m, formula = formula(m), data = data[indices, ], REML = FALSE, control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
        },
        warning = function(w) {
          if (grepl("convergence|eigenvalue", conditionMessage(w))) {  # Filter specific warnings
            #warning_counter <<- warning_counter + 1
          }
          invokeRestart("muffleWarning")
        }
      )
      
      #boot_model <- update(m, formula = formula(m), data = data[indices, ], REML = FALSE, control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
      
      # Check for singular fit
      if (isSingular(boot_model)) {
        cat(blue("singular fit \n"))
        singular_fit_counter <<- singular_fit_counter + 1
        return(rep(NA, nrow(pred.data)))  # Return NA if the model is singular
      }
      
       #Predict using the bootstrapped model
      predictions <- predict(boot_model, newdata = pred.data, type = "response", re.form = ~0)  
      
      if (link == "log") {
        boot_preds <- exp(predictions)
      } else if (link == "inverse") {
        boot_preds <- 1 / predictions
      } else if (link == "identity") {
        boot_preds <- predictions
      } else {
        stop("Unsupported link function. Use 'log', 'inverse', or 'identity'.")
      }
      return(boot_preds)
    }, R = nboots)
    
    # Calculate confidence intervals
    # Filter out NA predictions (from singular fits)
    valid_preds <- boots_results$t[complete.cases(boots_results$t), ]
    sim_preds <- t(valid_preds)
    lower_ci <- apply(sim_preds, 1, quantile, probs = 0.025) 
    upper_ci <- apply(sim_preds, 1, quantile, probs = 0.975) 
    print(lower_ci)
    print(upper_ci)
    
    # Print the number of warnings that occurred
    #cat("Number of convergence warnings during bootstrapping:", warning_counter, "out of", nboots, "bootstrap iterations.\n")
    # Print the number of warnings and singular fits that occurred
    #cat("Number of convergence warnings during bootstrapping:", warning_counter, "out of", nboots, "bootstrap iterations.\n")
    cat(blue("Number of singular fits during bootstrapping:", singular_fit_counter, "out of", nboots, "bootstrap iterations.\n"))
    
    # If keep.boots = TRUE, return the bootstrapped predictions along with the summary results
    if (keep.boots) {
      return(list(predictions = data.frame(pred.data, fit = prediction, lwr = lower_ci, upr = upper_ci),
                  boot_samples = boots_results$t))
    }
  } else {
    lower_ci <- upper_ci <- NULL
  }
  
  
  # If keep.boots = FALSE, return only the summary predictions
  return(data.frame(pred.data, fit = prediction, lwr = lower_ci, upr = upper_ci))
}
# 4. Function for bootstrappin glmmtmb
boot.ci.predict.glmmTMB <- function(m, data, pred.data, reqcol, centercol = NULL, nboots = NULL, keep.boots = FALSE) {
  
  
  # Initialize a counter singular fit
  #warning_counter <- 0
  singular_fit_counter <- 0
  
  # Filter pred.data to use only variables specified in 'reqcol'
  pred.data <- pred.data[ , reqcol, drop = FALSE]
  #print(pred.data)
  
  # Centering variables
  if (!is.null(centercol)) {
    for (var in centercol) {
      if (!is.numeric(data[[var]])) {
        data[[var]] <- as.numeric(as.factor(data[[var]]))  # Convert factor to numeric
      }
    }
    
    # Center the data
    centered_data <- scale(data[centercol], center = TRUE, scale = FALSE)
    means <- attr(centered_data, "scaled:center")  # Extract the mean values
    for (var in centercol) {
      pred.data[[var]] <- rep(means[var], nrow(pred.data))  # Repeat the mean value for each row
    }
  }
  
  # Predict values based on the model
  prediction <- predict(m, newdata = pred.data, type = "response", re.form = ~0)
  #print(prediction)
  
  # Perform bootstrapping to calculate confidence intervals
  if (!is.null(nboots)) {
    set.seed(123)
    boots_results <- boot(data, statistic = function(data, indices) {
      boot_model <- withCallingHandlers(
        {
          update(m, formula = formula(m), data = data[indices, ], family = family(m))
        },
        warning = function(w) {
          if (grepl("convergence|eigenvalue", conditionMessage(w))) {  # Filter specific warnings
            #warning_counter <<- warning_counter + 1
          }
          invokeRestart("muffleWarning")
        }
      )
      
      #Predict using the bootstrapped model
      
      predictions <- predict(boot_model, newdata = pred.data, type = "response", re.form = ~0)  
      
      return(predictions)
    }, R = nboots)
    
    # Calculate confidence intervals
    # Filter out NA predictions (from singular fits)
    valid_preds <- boots_results$t[complete.cases(boots_results$t), ]
    sim_preds <- t(valid_preds)
    lower_ci <- apply(sim_preds, 1, quantile, probs = 0.025) 
    upper_ci <- apply(sim_preds, 1, quantile, probs = 0.975) 
    print(lower_ci)
    print(upper_ci)
    
    # Print the number of warnings that occurred
    #cat("Number of convergence warnings during bootstrapping:", warning_counter, "out of", nboots, "bootstrap iterations.\n")
    # Print the number of warnings and singular fits that occurred
    #cat("Number of convergence warnings during bootstrapping:", warning_counter, "out of", nboots, "bootstrap iterations.\n")
    cat(blue("Number of singular fits during bootstrapping:", singular_fit_counter, "out of", nboots, "bootstrap iterations.\n"))
    
    # If keep.boots = TRUE, return the bootstrapped predictions along with the summary results
    if (keep.boots) {
      return(list(predictions = data.frame(pred.data, fit = prediction, lwr = lower_ci, upr = upper_ci),
                  boot_samples = boots_results$t))
    }
  } else {
    lower_ci <- upper_ci <- NULL
  }
  
  
  # If keep.boots = FALSE, return only the summary predictions
  return(data.frame(pred.data, fit = prediction, lwr = lower_ci, upr = upper_ci))
}



#  5. plotting function with horizontal fitted lines and vertical conf interval

factor.interaction.group.ci.plot <- function(plot.data, 
                                             coefs, 
                                             plot.data.col.x, 
                                             plot.data.col.y, 
                                             plot.data.col.group,
                                             x.labs, 
                                             y.labs, 
                                             x.discrete.labels, 
                                             x.group.pos,
                                             y.lim,
                                             div = NULL,
                                             scale.size.range = NULL,
                                             color.gradient.low = "grey90",   
                                             color.gradient.high = "grey30",  
                                             color.breaks = NULL,             
                                             size.breaks = NULL,              
                                             y.ax.transformation = FALSE,
                                             inv_y_transform_fun = NULL,
                                             y.breaks = NULL,
                                             size.scaling = NULL,
                                             aspect.ratio = NULL,
                                             title = NULL,
                                             legend = FALSE,
                                             p_value = NULL
) {
  
  # Access columns using [[ ]] notation
  x_column <- plot.data[[plot.data.col.x]]
  y_column <- plot.data[[plot.data.col.y]]
  group_column <- plot.data[[plot.data.col.group]]
  
  # Apply inverse transformation to the y-axis if required
  if (y.ax.transformation && !is.null(inv_y_transform_fun)) {
    y_column <- inv_y_transform_fun(y_column)
    coefs$fit <- inv_y_transform_fun(coefs$fit)
    coefs$lwr <- inv_y_transform_fun(coefs$lwr)
    coefs$upr <- inv_y_transform_fun(coefs$upr)
  }
  
  # Scale y values if needed
  if (!is.null(div)) {
    y_column <- y_column / div
    coefs$fit <- coefs$fit / div
    coefs$lwr <- coefs$lwr / div
    coefs$upr <- coefs$upr / div
  }
  
  
  if (!is.null(size.scaling)) {
    if (size.scaling == "log") {
      size_mapping <- log(plot.data$n + 1)
    } else if (size.scaling == "sqrt") {
      size_mapping <- sqrt(plot.data$n)
    } else if (size.scaling == "rescale") {
      size_mapping <- scales::rescale(plot.data$n, to = c(1, 5))
    } else {
      stop("Unsupported size scaling method. Use 'log', 'sqrt', or 'rescale'.")
    }
  } else {
    size_mapping <- plot.data$n  
  }
  
  
  # Create the base plot using dynamically accessed columns
  p <- ggplot(plot.data, aes(x = interaction(x_column), y = y_column)) +
    
    # Plot individual points with custom size scaling if given
    geom_point(aes(size = size_mapping, color = n), alpha = 0.3) +
  
    # Labels and minimal theme
  labs(x = x.labs, y = y.labs, title = title) +
    scale_x_discrete(labels = x.discrete.labels) +
    
    # Set color gradient dynamically
    scale_color_gradient(low = color.gradient.low, high = color.gradient.high, breaks = color.breaks) +
    
    # Scale y-axis limits and breaks
    scale_y_continuous(limits = y.lim, breaks = y.breaks) + 
    
    # Theme adjustments
    theme_minimal() +
    theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 1.5),  
          panel.grid.major.x = element_blank(),                                      
          panel.grid.major.y = element_line(color = "grey80", linewidth = 0.5),      
          panel.grid.minor = element_blank(),                                        
          axis.line = element_line(color = "black", linewidth = 0.5),
          plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
          aspect.ratio = aspect.ratio) 

          
  
  if (legend){
  p <- p + guides(
    color = guide_legend("n"),  
    size = guide_legend("n")
  )
  }
  else {
    p <- p + guides(
      color = "none",  
      size = "none"
    )
  }

    # Adjusted size range and breaks for user-specified `n` range
  p <- p + scale_size_continuous(range = scale.size.range, breaks = size.breaks)
  # Generate positions on the x-axis corresponding to the groups
  x_positions <- x.group.pos
  
  # Add the fitted values as horizontal lines
  p <- p + annotate("segment", x = x_positions - 0.2, xend = x_positions + 0.2, 
                    y = coefs$fit, yend = coefs$fit, color = "blue", size = 0.2)
  
  # Add vertical error bars (confidence intervals)
  p <- p + annotate("segment", x = x_positions, xend = x_positions, 
                    y = coefs$lwr, yend = coefs$upr, color = "blue", size = 0.2)
  
  # Add horizontal whiskers at the bottom of the vertical error bars
  p <- p + annotate("segment", x = x_positions - 0.05, xend = x_positions + 0.05, 
                    y = coefs$lwr, yend = coefs$lwr, color = "blue", size = 0.2)
  
  # Add horizontal whiskers at the top of the vertical error bars
  p <- p + annotate("segment", x = x_positions - 0.05, xend = x_positions + 0.05, 
                    y = coefs$upr, yend = coefs$upr, color = "blue", size = 0.2)
  
  # Connect points for each parent where id_parent is the same across registers
  p <- p + geom_line(data = plot.data, aes(x = interaction(x_column), group = group_column), 
                     color = "lightgrey", linetype = "solid", linewidth = 0.3, alpha = 0.3)
  
  # Adding p-value
  p <- p + annotate("text", x = max(x_positions), y = max(y.lim), label = p_value, 
               size = 5, color = "blue", fontface = "italic", hjust = 0.6, vjust = 1) 
  return(p)
}

# 6. compute contrasts function
compute_contrasts <- function(margian_means, compute_pairwise = FALSE, custom_contrasts = NULL, ...) {
  
  # Apply contrasts
  if (!is.null(custom_contrasts)) {
    # Use custom contrasts provided by the user
    contrast_results <- emmeans::contrast(margian_means, method = custom_contrasts)
  } else if (compute_pairwise) {
    # Compute default pairwise contrasts
    contrast_results <- emmeans::contrast(margian_means, method = "pairwise")
  } else {
    # No contrasts computed
    contrast_results <- NULL
  }
  print(contrast_results)
  # Return results as a list
  return(list(contrasts = contrast_results))
}



lmer.plot.emmeans.contrasts <- function(emmeans.data, 
                                   contrasts.data, 
                                   raw_data, 
                                   y.ax.transformation = FALSE, 
                                   inv_y_transform_fun = NULL, 
                                   div = NULL,
                                   x_labs = NULL, 
                                   y_labs = NULL, 
                                   y_lim = NULL,
                                   scale_size_range = NULL,
                                   color_gradient_low = "grey60",   
                                   color_gradient_high = "grey10",  
                                   color_breaks = NULL,             
                                   size_breaks = NULL,              
                                   y_breaks = NULL,
                                   size_scaling = NULL,
                                   aspect_ratio = NULL,
                                   title = NULL,
                                   legend = FALSE
) {
  # Load required packages
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    install.packages("ggplot2")
  }
  library(ggplot2)
  
  # Convert emmeans_data to data frame if necessary
  if (!is.data.frame(emmeans.data)) {
    emmeans_data <- as.data.frame(emmeans.data)
  }
  
  # Create Group variable in emmeans_data
  emmeans_data$Group <- interaction(emmeans_data$Register, emmeans_data$Language, sep = "*")
  
  # Ensure Group is a factor with a specific order
  emmeans_data$Group <- factor(emmeans_data$Group, levels = unique(emmeans_data$Group))
  
  # Print levels of Group for debugging
  #print("Levels of emmeans_data$Group:")
  #print(levels(emmeans_data$Group))
  
  # Convert contrasts_data to data frame if necessary
  if (!is.data.frame(contrasts.data)) {
    contrasts_data <- as.data.frame(contrasts.data)
  }
  
  # Prepare comparisons and annotations based on contrasts_data
  comparisons <- list()
  annotations <- c()
  
  for (i in 1:nrow(contrasts_data)) {
    contrast <- contrasts_data$contrast[i]
    # Remove spaces and split contrast into groups
    contrast_clean <- gsub(" ", "", contrast)
    # Split on "-" or "vs"
    groups <- unlist(strsplit(contrast_clean, "-|vs"))
    comparisons[[i]] <- groups
    contrast <- contrasts_data$contrast[i]
    p_value <- contrasts_data$p.value[i]
    # Format p-value
    p_value_formatted <- ifelse(p_value < 0.001, "< 0.001", sprintf("= %.3f", p_value))
    annotations[i] <- paste0("p ", p_value_formatted)#,  " (", contrast, ")")
    
    # Debugging statements
    #cat("Processing contrast:", contrast, "\n")
    #cat("Extracted groups:", groups, "\n")
  }
  
  # Create the plot
  p <- ggplot()
  
  # If raw_data is provided, plot raw data points
  if (!is.null(raw_data)) {
    # Ensure raw_data has Register, Language, and value columns
    if (!all(c("Register", "Language", "value") %in% colnames(raw_data))) {
      stop("raw_data must contain 'Register', 'Language', and 'value' columns.")
    }
    # Create Group variable in raw_data
    raw_data$Group <- interaction(raw_data$Register, raw_data$Language, sep = "*")
    # Ensure Group levels match emmeans_data
    raw_data$Group <- factor(raw_data$Group, levels = levels(emmeans_data$Group))
  }
  
  
  # Apply inverse transformation to the y-axis if required
  if (y.ax.transformation && !is.null(inv_y_transform_fun)) {
    raw_data$value <- inv_y_transform_fun(raw_data$value)
    emmeans_data$emmean <- inv_y_transform_fun(emmeans_data$emmean)
    emmeans_data$lower.CL <- inv_y_transform_fun(emmeans_data$lower.CL)
    emmeans_data$upper.CL <- inv_y_transform_fun(emmeans_data$upper.CL)
  }
  
  # Scale y values if needed
  if (!is.null(div)) {
    raw_data$value <- raw_data$value / div
    emmeans_data$emmean <- emmeans_data$emmean / div
    emmeans_data$lower.CL <- emmeans_data$lower.CL / div
    emmeans_data$upper.CL <- emmeans_data$upper.CL / div
  }
  
  # Adjust point sizes based on the number of observations
  if (!is.null(size_scaling)) {
    if (size_scaling == "log") {
      size_mapping <- log(raw_data$n + 1)
    } else if (size_scaling == "sqrt") {
      size_mapping <- sqrt(raw_data$n)
    } else if (size_scaling == "rescale") {
      size_mapping <- scales::rescale(raw_data$n, to = c(1, 5))
    } else {
      stop("Unsupported size scaling method. Use 'log', 'sqrt', or 'rescale'.")
    }
  } else {
    size_mapping <- raw_data$n  
  }
  
  # Plot raw data points
  if (!is.null(raw_data)) {
    p <- p + geom_point(data = raw_data, aes(x = Group, y = value, size = size_mapping), alpha = 0.3, color = "gray50")
  }
  # Set size range, color gradients, and axis limits
  p <- p + scale_size_continuous(range = scale_size_range, breaks = size_breaks) +
    scale_color_gradient(low = color_gradient_low, high = color_gradient_high, breaks = color_breaks) +
    scale_y_continuous(limits = y_lim, breaks = y_breaks)
  
  if (legend) {
    p <- p + guides(
      color = guide_legend("n"),  
      size = guide_legend("n")
    )
  } else {
    p <- p + guides(color = "none", size = "none")
  }
  
  # Plot estimated marginal means as points and lines
  p <- p + 
    geom_point(data = emmeans_data, aes(x = Group, y = emmean), size = 3, color = "blue") +
    geom_errorbar(data = emmeans_data, aes(x = Group, ymin = lower.CL, ymax = upper.CL),
                  width = 0.1, color = "blue")
  
  # Add significance annotations with lines and whiskers
  if (length(comparisons) > 0) {
    # Adjust y positions for annotations
    max_y <- max(raw_data$value)
    y_positions <- seq(max_y + ((y_breaks[2]-y_breaks[1])/10), by = ((y_breaks[2]-y_breaks[1])/2), length.out = length(comparisons))
    
    # Create empty data frames to collect segments and annotations
    segment_data <- data.frame(x1 = numeric(), x2 = numeric(), y = numeric())
    annotation_data <- data.frame(x = numeric(), y = numeric(), label = character())
    
    # Create a set to store pairs that have already been annotated
    annotated_pairs <- list()
    
    for (i in 1:length(comparisons)) {
      group1 <- comparisons[[i]][1]
      group2 <- comparisons[[i]][2]
      
      # Get x positions of the groups (convert to numeric as factors are used)
      x1 <- as.numeric(which(levels(emmeans_data$Group) == group1))
      x2 <- as.numeric(which(levels(emmeans_data$Group) == group2))
      
      # Ensure x1 is less than x2
      if (x1 > x2) {
        temp <- x1
        x1 <- x2
        x2 <- temp
      }
      
      # Combine the groups into a string to use as a key
      pair_key <- paste(sort(c(group1, group2)), collapse = "-")
      
      # Check if this pair has already been annotated
      if (pair_key %in% annotated_pairs) {
        next  # Skip if the pair has already been annotated
      } else {
        annotated_pairs <- c(annotated_pairs, pair_key)  # Mark this pair as annotated
      }
      
      # Set y position for the annotation
      y_position <- y_positions[i]
      
      # Add the data for the line segment (horizontal line and whiskers)
      segment_data <- rbind(segment_data, data.frame(x1 = x1, x2 = x2, y = y_position))
      
      # Add the data for the p-value annotation
      annotation_data <- rbind(annotation_data, data.frame(x = mean(c(x1, x2)), y = y_position, label = annotations[i]))
    }
    
    # Now, add the segments and annotations to the plot at once
    p <- p +
      geom_segment(data = segment_data, aes(x = x1, xend = x2, y = y, yend = y), inherit.aes = FALSE, color = "black") +
      geom_segment(data = segment_data, aes(x = x1, xend = x1, y = y, yend = y - ((y_breaks[2]-y_breaks[1])/10)), inherit.aes = FALSE, color = "black") +  # Left whisker
      geom_segment(data = segment_data, aes(x = x2, xend = x2, y = y, yend = y - ((y_breaks[2]-y_breaks[1])/10)), inherit.aes = FALSE, color = "black") +  # Right whisker
      geom_text(data = annotation_data, aes(x = x, y = y + ((y_breaks[2]-y_breaks[1])/20), label = label), inherit.aes = FALSE, vjust = 0, hjust = 0.5)
    
    # Connect points for the same `spkid`
    p <- p + geom_line(data = raw_data, aes(x = Group, y = value, group = spkid), color = "lightgrey", linetype = "solid", linewidth = 0.3, alpha = 0.3)
    
    # Customize the plot
    p <- p +
      labs(x = x_labs, y = y_labs, title = title) +
      theme_minimal() +
      theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 1.5),  
            panel.grid.major.x = element_blank(),                                      
            panel.grid.major.y = element_line(color = "grey80", linewidth = 0.5),      
            panel.grid.minor = element_blank(),
            axis.text.x = element_text(angle = 45, hjust = 1),
            axis.line = element_line(color = "black", linewidth = 0.5),
            plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
            aspect.ratio = aspect_ratio) 
    
    # Display the plot
    print(p)
  }
}

glmmTB.plot.emmeans.contrasts <- function(emmeans.data, 
                                        contrasts.data, 
                                        raw_data, 
                                        x_labs = NULL, 
                                        y_labs = NULL, 
                                        y_lim = NULL,
                                        scale_size_range = NULL,
                                        color_gradient_low = "grey60",   
                                        color_gradient_high = "grey10",  
                                        color_breaks = NULL,             
                                        size_breaks = NULL,              
                                        y_breaks = NULL,
                                        size_scaling = NULL,
                                        aspect_ratio = NULL,
                                        title = NULL,
                                        legend = FALSE
) {
  # Load required packages
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    install.packages("ggplot2")
  }
  library(ggplot2)
  
  # Convert emmeans_data to data frame if necessary
  if (!is.data.frame(emmeans.data)) {
    emmeans_data <- as.data.frame(emmeans.data)
  }
  
  # Create Group variable in emmeans_data
  emmeans_data$Group <- interaction(emmeans_data$Register, emmeans_data$Language, sep = "*")
  
  # Ensure Group is a factor with a specific order
  emmeans_data$Group <- factor(emmeans_data$Group, levels = unique(emmeans_data$Group))
  
  # Print levels of Group for debugging
  #print("Levels of emmeans_data$Group:")
  #print(levels(emmeans_data$Group))
  
  # Convert contrasts_data to data frame if necessary
  if (!is.data.frame(contrasts.data)) {
    contrasts_data <- as.data.frame(contrasts.data)
  }
  
  # Prepare comparisons and annotations based on contrasts_data
  comparisons <- list()
  annotations <- c()
  
  for (i in 1:nrow(contrasts_data)) {
    contrast <- contrasts_data$contrast[i]
    # Remove spaces and split contrast into groups
    contrast_clean <- gsub(" ", "", contrast)
    # Split on "-" or "vs"
    groups <- unlist(strsplit(contrast_clean, "-|vs"))
    comparisons[[i]] <- groups
    contrast <- contrasts_data$contrast[i]
    p_value <- contrasts_data$p.value[i]
    # Format p-value
    p_value_formatted <- ifelse(p_value < 0.001, "< 0.001", sprintf("= %.3f", p_value))
    annotations[i] <- paste0("p ", p_value_formatted)#,  " (", contrast, ")")
    
    # Debugging statements
    #cat("Processing contrast:", contrast, "\n")
    #cat("Extracted groups:", groups, "\n")
  }
  
  # Create the plot
  p <- ggplot()
  
  # If raw_data is provided, plot raw data points
  if (!is.null(raw_data)) {
    # Ensure raw_data has Register, Language, and value columns
    if (!all(c("Register", "Language", "value") %in% colnames(raw_data))) {
      stop("raw_data must contain 'Register', 'Language', and 'value' columns.")
    }
    # Create Group variable in raw_data
    raw_data$Group <- interaction(raw_data$Register, raw_data$Language, sep = "*")
    # Ensure Group levels match emmeans_data
    raw_data$Group <- factor(raw_data$Group, levels = levels(emmeans_data$Group))
  }
  
  
  # Adjust point sizes based on the number of observations
  if (!is.null(size_scaling)) {
    if (size_scaling == "log") {
      size_mapping <- log(raw_data$n + 1)
    } else if (size_scaling == "sqrt") {
      size_mapping <- sqrt(raw_data$n)
    } else if (size_scaling == "rescale") {
      size_mapping <- scales::rescale(raw_data$n, to = c(1, 5))
    } else {
      stop("Unsupported size scaling method. Use 'log', 'sqrt', or 'rescale'.")
    }
  } else {
    size_mapping <- raw_data$n  
  }
  
  # Plot raw data points
  if (!is.null(raw_data)) {
    p <- p + geom_point(data = raw_data, aes(x = Group, y = value, size = size_mapping), alpha = 0.3, color = "gray50")
  }
  # Set size range, color gradients, and axis limits
  p <- p + scale_size_continuous(range = scale_size_range, breaks = size_breaks) +
    scale_color_gradient(low = color_gradient_low, high = color_gradient_high, breaks = color_breaks) +
    scale_y_continuous(limits = y_lim, breaks = y_breaks)
  
  if (legend) {
    p <- p + guides(
      color = guide_legend("n"),  
      size = guide_legend("n")
    )
  } else {
    p <- p + guides(color = "none", size = "none")
  }
  
  # Plot estimated marginal means as points and lines
  p <- p + 
    geom_point(data = emmeans_data, aes(x = Group, y = response), size = 3, color = "blue") +
    geom_errorbar(data = emmeans_data, aes(x = Group, ymin = asymp.LCL, ymax = asymp.UCL),
                  width = 0.1, color = "blue")
  
  # Add significance annotations with lines and whiskers
  if (length(comparisons) > 0) {
    # Adjust y positions for annotations
    max_y <- max(raw_data$value)
    y_positions <- seq(max_y + (y_breaks[2]-y_breaks[1]), by = ((y_breaks[2]-y_breaks[1])/2), length.out = length(comparisons))
    
    # Create empty data frames to collect segments and annotations
    segment_data <- data.frame(x1 = numeric(), x2 = numeric(), y = numeric())
    annotation_data <- data.frame(x = numeric(), y = numeric(), label = character())
    
    # Create a set to store pairs that have already been annotated
    annotated_pairs <- list()
    
    for (i in 1:length(comparisons)) {
      group1 <- comparisons[[i]][1]
      group2 <- comparisons[[i]][2]
      
      # Get x positions of the groups (convert to numeric as factors are used)
      x1 <- as.numeric(which(levels(emmeans_data$Group) == group1))
      x2 <- as.numeric(which(levels(emmeans_data$Group) == group2))
      
      # Ensure x1 is less than x2
      if (x1 > x2) {
        temp <- x1
        x1 <- x2
        x2 <- temp
      }
      
      # Combine the groups into a string to use as a key
      pair_key <- paste(sort(c(group1, group2)), collapse = "-")
      
      # Check if this pair has already been annotated
      if (pair_key %in% annotated_pairs) {
        next  # Skip if the pair has already been annotated
      } else {
        annotated_pairs <- c(annotated_pairs, pair_key)  # Mark this pair as annotated
      }
      
      # Set y position for the annotation
      y_position <- y_positions[i]
      
      # Add the data for the line segment (horizontal line and whiskers)
      segment_data <- rbind(segment_data, data.frame(x1 = x1, x2 = x2, y = y_position))
      
      # Add the data for the p-value annotation
      annotation_data <- rbind(annotation_data, data.frame(x = mean(c(x1, x2)), y = y_position, label = annotations[i]))
    }
    
    # Now, add the segments and annotations to the plot at once
    p <- p +
      geom_segment(data = segment_data, aes(x = x1, xend = x2, y = y, yend = y), inherit.aes = FALSE, color = "black") +
      geom_segment(data = segment_data, aes(x = x1, xend = x1, y = y, yend = y - ((y_breaks[2]-y_breaks[1])/4)), inherit.aes = FALSE, color = "black") +  # Left whisker
      geom_segment(data = segment_data, aes(x = x2, xend = x2, y = y, yend = y - ((y_breaks[2]-y_breaks[1])/4)), inherit.aes = FALSE, color = "black") +  # Right whisker
      geom_text(data = annotation_data, aes(x = x, y = y + ((y_breaks[2]-y_breaks[1])/20), label = label), inherit.aes = FALSE, vjust = 0, hjust = 0.5)
    
    # Connect points for the same `spkid`
    p <- p + geom_line(data = raw_data, aes(x = Group, y = value, group = spkid), color = "lightgrey", linetype = "solid", linewidth = 0.3, alpha = 0.3)
    
    # Customize the plot
    p <- p +
      labs(x = x_labs, y = y_labs, title = title) +
      theme_minimal() +
      theme(panel.border = element_rect(color = "black", fill = NA, linewidth = 1.5),  
            panel.grid.major.x = element_blank(),                                      
            panel.grid.major.y = element_line(color = "grey80", linewidth = 0.5),      
            panel.grid.minor = element_blank(),
            axis.text.x = element_text(angle = 45, hjust = 1),
            axis.line = element_line(color = "black", linewidth = 0.5),
            plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
            aspect.ratio = aspect_ratio) 
    
    # Display the plot
    print(p)
  }
}
