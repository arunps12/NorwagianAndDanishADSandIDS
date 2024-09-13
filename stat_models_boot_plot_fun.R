
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
                                           ...) {
  
  # Combine input dataframes
  data_list <- list(...)
  combined_data <- bind_rows(data_list)
  cat(blue("structure of combined data. \n"))
  str(combined_data)
  
  
  
  # Extract the response variable 
  terms_obj <- terms(fullmformula)
  response_var <- as.character(attr(terms_obj, "variables"))[2]  # Extract the response variable
  
  # Standardize all numeric columns except the response variable
  # Identify numeric columns
  numeric_columns <- sapply(combined_data, is.numeric)
  
  # Subset numeric columns and exclude the response variable
  numeric_columns <- setdiff(names(combined_data)[which(numeric_columns)], response_var)
  
  combined_data[numeric_columns] <- lapply(combined_data[numeric_columns], scale)  # Standardize numeric columns
  
  # Check structure of combined data after scaling
  cat(blue("structure of combined data after scaling. \n"))
  str(combined_data)
  
  
  # Extract fixed effects
  fixed_effects <- attr(terms_obj, "term.labels")  
  
  # Extract the random effects
  random_effects <- findbars(fullmformula)  
  random_effects_str <- paste(sapply(random_effects, deparse)) # Convert random effects to string
  
  # Remove random effects from the fixed effects list
  fixed_effects_clean <- fixed_effects[!fixed_effects %in% random_effects_str]
  
  # Add parentheses around the each random effects part
  formatted_random_effects <- sapply(random_effects, function(re) {
    re_str <- deparse(re)
    if (!grepl("^\\(", re_str)) {
      re_str <- paste0("(", re_str, ")")
    }
    return(re_str)
  })
  random_effects_str <- paste(formatted_random_effects, collapse = " + ")
  
  # Histogram of Response variable
  p_histogram <- ggplot(combined_data, aes_string(x = response_var)) +
    geom_histogram(bins = 30) +
    labs(title = "Histogram of Response Variable")
  print(p_histogram)  
  
  # Check if the response variable is normally distributed
  if (shapiro.test(combined_data[[response_var]])$p.value < 0.05) {
    combined_data[[response_var]] <- log(combined_data[[response_var]] + 1)
    cat(blue("Response variable is not normally distributed. Log transformation applied.\n"))
    cat(blue("structure of combined data after transformed. \n"))
    str(combined_data)
    
    # Histogram of Response variable
    p_histogram <- ggplot(combined_data, aes_string(x = response_var)) +
      geom_histogram(bins = 30) +
      labs(title = "Histogram of log transformed Response Variable")
    print(p_histogram)  
  } else {
    cat(blue("Response variable is normally distributed, hence, no transformation applied.\n"))
  }
  
  # fitting the linear mixed model with provided optimizer and maxfun
  if (fullm){
  fit_model <- tryCatch({
    lmer(fullmformula, data = combined_data, REML = FALSE, 
         control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
  }, error = function(e) {
    #cat(blue("Error in fitting REML False model:", e$message, "\n"))
    return(NULL)
  })
  
  # Check if the full model was fitted successfully
  if (is.null(fit_model)) {
    cat(blue("Full model failed. Trying to fit the reduced model if provided.\n"))
    cat(blue("Model did not fit. Consider using a different optimizer or increasing maxfun.\n"))
    cat(blue("Available optimizers include: 'bobyqa', 'nloptwrap', 'optim', 'nlopt', etc.\n"))
  }
  
  fit_model_REML <- lmerTest::lmer(fullmformula, data = combined_data, REML = TRUE, 
                                   control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
  
  #if (is.null(fit_model)) {
    #cat(blue("Model did not fit. Consider using a different optimizer or increasing maxfun.\n"))
    #cat(blue("Available optimizers include: 'bobyqa', 'nloptwrap', 'optim', 'nlopt', etc.\n"))
  #}
  
  # Check for singular fit
  if (isSingular(fit_model)) {
    cat(blue("Model is singular. Try simplifying the random effects structure by removing some random slopes or interactions.\n"))
  }
  
  logLik_full <- logLik(fit_model)
  df_full <- attr(logLik_full, "df")
  cat(blue("Full Model: Log-Likelihood =", round(as.numeric(logLik_full), 2), 
           ", Degrees of Freedom =", df_full, "\n"))
  vif <- vif(fit_model)
  cat(blue("max VIF values of the full model:", max(vif), "\n"))
  }
  
  # fitting the linear mixed model with reduced model
  if (redm && !is.null(redmformula)) {
    red_fit_model <- tryCatch({
      lmer(redmformula, data = combined_data, REML = FALSE, 
           control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
    }, error = function(e) {
      cat(blue("Error in fitting the reduced model: ", e$message, "\n"))
      return(NULL)
    })
    
    red_fit_model_REML <- lmerTest::lmer(redmformula, data = combined_data, REML = TRUE, 
                                         control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
    
    if (is.null(red_fit_model)) {
      cat(blue("Reduced Model did not fit. Consider using a different optimizer or increasing maxfun.\n"))
      cat(blue("Available optimizers include: 'bobyqa', 'nloptwrap', 'optim', 'nlopt', etc.\n"))
    }
    
    # Check for singular fit
    if (isSingular(red_fit_model)) {
      cat(blue("Reduced Model is also singular fit. Try simplifying the random effects structure by removing some random slopes or interactions.\n"))
    }
    logLik_reduced <- logLik(red_fit_model)
    df_reduced <- attr(logLik_reduced, "df") 
    cat(blue("Reduced Model: Log-Likelihood =", round(as.numeric(logLik_reduced), 2), 
             ", Degrees of Freedom =", df_reduced, "\n"))
    
    # Extract residuals and fitted values of reduced model
    residuals <- resid(red_fit_model)
    fitted_values <- fitted(red_fit_model)
    
    # Plot residuals vs fitted values
    p_residuals_vs_fitted <- ggplot(data = data.frame(fitted = fitted_values, residuals = residuals), aes(x = fitted, y = residuals)) +
      geom_point() +
      geom_smooth(method = "lm") +
      labs(title = "Residuals vs Fitted Values of reduced model ", x = "Fitted Values", y = "Residuals")
    print(p_residuals_vs_fitted)  
    
    # Q-Q plot of residuals
    p_qq_plot <- ggplot(data = data.frame(residuals = residuals), aes(sample = residuals)) +
      geom_qq() +
      geom_qq_line() +
      labs(title = "Q-Q Plot of Residuals for reduced model")
    print(p_qq_plot)  
    
    # Extract model summary
    red_model_summary <- summary(red_fit_model_REML)
    cat(blue("Summary of the reduced model:\n"))
    print(red_model_summary)
    red_vif <- vif(red_fit_model)
    cat(blue("VIF values of the reduced model:", max(red_vif), "\n"))
    #print(red_vif)
    if (nullm && !is.null(nullmformula)) {
      null_fit_model <- tryCatch({
        lmer(nullmformula, data = combined_data, REML = FALSE, 
             control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
      }, error = function(e) {
        cat(blue("Error in fitting REML False model:", e$message, "\n"))
        return(NULL)
      })
      cat(blue("Compare reduced  and null models using ANOVA.\n"))
      reduced_null_model_comparison <- anova(red_fit_model, null_fit_model)
      print(reduced_null_model_comparison)
    }
    # Return the fitted model, data, optimizer , and maxfun
    cat(blue("Return the reduced fitted model, data, optimizer , and maxfun. \n"))
    return(list(model = red_fit_model, fit_data = combined_data, optimizer = optimizer, maxfun = maxfun))
  } else {
    
    cat(blue("No reduced model provided, proceeding with full model.\n"))
    
    # Extract residuals and fitted values of full model
    residuals <- resid(fit_model)
    fitted_values <- fitted(fit_model)
    
    # Plot residuals vs fitted values for full model
    p_residuals_vs_fitted <- ggplot(data = data.frame(fitted = fitted_values, residuals = residuals), aes(x = fitted, y = residuals)) +
      geom_point() +
      geom_smooth(method = "lm") +
      labs(title = "Residuals vs Fitted Values for full model", x = "Fitted Values", y = "Residuals")
    print(p_residuals_vs_fitted)  
    
    # Q-Q plot of residuals for full model
    p_qq_plot <- ggplot(data = data.frame(residuals = residuals), aes(sample = residuals)) +
      geom_qq() +
      geom_qq_line() +
      labs(title = "Q-Q Plot of Residuals for full model")
    print(p_qq_plot)  
    
    # Extract model summary for full model
    model_summary <- summary(fit_model_REML)
    cat(blue("Summary of the full model:\n"))
    print(model_summary)
    if (nullm && !is.null(nullmformula)) {
      null_fit_model <- tryCatch({
        lmer(nullmformula, data = combined_data, REML = FALSE, 
             control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = maxfun)))
      }, error = function(e) {
        cat(blue("Error in fitting REML False model:", e$message, "\n"))
        return(NULL)
      })
      cat(blue("Compare full and null models using ANOVA.\n"))
      full_null_model_comparison <- anova(fit_model, null_fit_model)
      print(full_null_model_comparison)
    }
    
    # Return the fitted model, data, optimizer , and maxfun
    cat(blue("Return the full fitted model, data, optimizer , and maxfun. \n"))
    return(list(model = fit_model, fit_data = combined_data, optimizer = optimizer, maxfun = maxfun))
  }
}

# 2. fun for bootstrapped predictions

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
  predictions <- predict(m, newdata = pred.data, type = "response", re.form = ~0)
  
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
      predictions <- predict(boot_model, newdata = pred.data, re.form = ~0)
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
    
    # Print the number of warnings that occurred
    #cat("Number of convergence warnings during bootstrapping:", warning_counter, "out of", nboots, "bootstrap iterations.\n")
    # Print the number of warnings and singular fits that occurred
    #cat("Number of convergence warnings during bootstrapping:", warning_counter, "out of", nboots, "bootstrap iterations.\n")
    cat(blue("Number of singular fits during bootstrapping:", singular_fit_counter, "out of", nboots, "bootstrap iterations.\n"))
    
    # If keep.boots = TRUE, return the bootstrapped predictions along with the summary results
    if (keep.boots) {
      return(list(predictions = data.frame(pred.data, fit = predictions, lwr = lower_ci, upr = upper_ci),
                  boot_samples = boots_results$t))
    }
  } else {
    lower_ci <- upper_ci <- NULL
  }
  
  # If keep.boots = FALSE, return only the summary predictions
  return(data.frame(pred.data, fit = predictions, lwr = lower_ci, upr = upper_ci))
}



#  3. plotting function with horizontal fitted lines and vertical conf interval

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
                                             legend = FALSE
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
                    y = coefs$fit, yend = coefs$fit, color = "blue", size = 1.5)
  
  # Add vertical error bars (confidence intervals)
  p <- p + annotate("segment", x = x_positions, xend = x_positions, 
                    y = coefs$lwr, yend = coefs$upr, color = "blue", size = 1.5)
  
  # Add horizontal whiskers at the bottom of the vertical error bars
  p <- p + annotate("segment", x = x_positions - 0.05, xend = x_positions + 0.05, 
                    y = coefs$lwr, yend = coefs$lwr, color = "blue", size = 1)
  
  # Add horizontal whiskers at the top of the vertical error bars
  p <- p + annotate("segment", x = x_positions - 0.05, xend = x_positions + 0.05, 
                    y = coefs$upr, yend = coefs$upr, color = "blue", size = 1)
  
  # Connect points for each parent where id_parent is the same across registers
  p <- p + geom_line(data = plot.data, aes(x = interaction(x_column), group = group_column), 
                     color = "lightgrey", linetype = "solid", linewidth = 0.3, alpha = 0.3)  
  return(p)
}



