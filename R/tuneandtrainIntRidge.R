#' Tune and Train Internal Ridge 
#'
#' This function tunes and trains a Ridge classifier using the \code{glmnet} package. The function 
#' evaluates a sequence of lambda (regularization) values using internal cross-validation and selects 
#' the best model based on the Area Under the Curve (AUC).
#'
#' The function trains a logistic Ridge regression model on the training dataset and performs cross-validation 
#' to select the best lambda value. The lambda value that gives the highest AUC on the training dataset during 
#' cross-validation is chosen as the best model.
#' 
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables.
#' @param maxit An integer specifying the maximum number of iterations. Default is 120000.
#' @param nlambda An integer specifying the number of lambda values to use in the Ridge model. Default is 200.
#' @param nfolds An integer specifying the number of folds for cross-validation. Default is 5.
#' @param seed An integer specifying the random seed for reproducibility. Default is 123.
#'
#' @return A list containing the best lambda value (`best_lambda`) and the final trained model (`best_model`).
#' @export
#'
#' @examples
#' # Load sample data
#' data(sample_data_train)
#'
#' # Example usage
#' result <- tuneandtrainIntRidge(sample_data_train, maxit = 120000, 
#'   nlambda = 200, nfolds = 5, seed = 123)
#' result$best_lambda
#' result$best_model
#' 
tuneandtrainIntRidge <- function(data, maxit = 120000, nlambda = 200, nfolds = 5, seed = 123) {
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  
  # Set random seed for reproducibility
  set.seed(seed)
  
  # Split data into predictors and response
  X <- as.matrix(data[, -1])
  y <- as.factor(data[, 1])
  
  # Fit initial Ridge Regression model to obtain lambda sequence
  fit_Ridge <- glmnet::glmnet(x = X, y = y, family = "binomial", maxit = maxit, 
                              nlambda = nlambda, alpha = 0, standardize = TRUE)
  lamseq <- fit_Ridge$lambda
  
  # Cross-validation
  partition <- sample(rep(1:nfolds, length.out = nrow(data)))
  AUC_CV <- matrix(NA, nrow = length(lamseq), ncol = nfolds)
  
  for (j in 1:nfolds) {
    XTrain <- X[partition != j, , drop = FALSE]
    yTrain <- y[partition != j]
    XTest <- X[partition == j, , drop = FALSE]
    yTest <- y[partition == j]
    
    if (length(unique(yTest)) == 1) {
      AUC_CV[, j] <- NA
    } else {
      fit_Ridge_CV <- glmnet::glmnet(x = XTrain, y = yTrain, family = "binomial", 
                                     maxit = maxit, lambda = lamseq, alpha = 0, standardize = TRUE)
      pred_Ridge_CV <- stats::predict(fit_Ridge_CV, newx = XTest, s = lamseq, type = "response")
      
      for (i in 1:ncol(pred_Ridge_CV)) {
        AUC_CV[i, j] <- pROC::auc(response = yTest, predictor = as.numeric(pred_Ridge_CV[, i]), quiet = TRUE)
      }
    }
  }
  
  # Determine the best lambda based on the highest average AUC
  mean_AUC <- rowMeans(AUC_CV, na.rm = TRUE)
  best_lambda_idx <- which.max(mean_AUC)
  best_lambda <- lamseq[best_lambda_idx]
  
  # Final model training with the best lambda
  final_model <- glmnet::glmnet(x = X, y = y, family = "binomial", maxit = maxit, 
                                lambda = best_lambda, alpha = 0, standardize = TRUE)
  
  # Return the result
  res <- list(
    best_lambda = best_lambda,
    best_model = final_model
  )
  
  # Set class
  class(res) <- "IntRidge"
  return(res)
}
