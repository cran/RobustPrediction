#' Tune and Train Internal Boosting
#'
#' This function tunes and trains a Boosting classifier using the \code{mboost} package. The function 
#' evaluates a sequence of boosting iterations on the training dataset using internal cross-validation 
#' and selects the best model based on the Area Under the Curve (AUC).
#'
#' This function performs K-fold cross-validation on the training dataset, where the number of boosting 
#' iterations (\code{mstop}) is tuned to maximize the AUC. The optimal number of boosting iterations is selected, 
#' and the final model is trained on the entire training dataset.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables.
#' @param mstop_seq A numeric vector of boosting iterations to be evaluated. Default is a sequence 
#'   from 5 to 1000 with a step of 5.
#' @param nu A numeric value for the learning rate. Default is 0.1.
#'
#' @return A list containing the best number of boosting iterations (`best_mstop`) and
#'   the final Boosting classifier model (`best_model`).
#' 
#' @export
#'
#' @examples
#' \donttest{
#' # Load sample data
#' data(sample_data_train)
#'
#' # Example usage
#' mstop_seq <- seq(5, 5000, by = 5)
#' result <- tuneandtrainIntBoost(sample_data_train, mstop_seq, nu = 0.1)
#' result$best_mstop
#' result$best_model
#' }
tuneandtrainIntBoost <- function(data, mstop_seq = seq(5, 1000, by = 5), nu = 0.1) {
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  
  Train <- data
  
  # Fit initial boosting model using mboost package
  fit_Boost <- mboost::glmboost(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]),
                                family = mboost::Binomial(), 
                                control = mboost::boost_control(mstop = max(mstop_seq), nu = nu),
                                center = FALSE)
  
  # Split Train in 5 parts for cross-validation
  K <- 5
  n <- nrow(Train)
  partition <- rep(1:K, length = n)
  partition <- partition[sample(n)]
  
  # Cross Validation
  AUC_CV <- matrix(NA, nrow = length(mstop_seq), ncol = K)
  
  for (j in 1:K) {
    XTrain <- Train[partition != j, ]
    XTest <- Train[partition == j, ]
    
    if (length(levels(as.factor(XTest[, 1]))) == 1) {
      AUC_CV[, j] <- NA
    } else {
      # Fit Boosting Model using mboost package
      fit_Boost_CV <- mboost::glmboost(x = as.matrix(XTrain[, -1]), y = as.factor(XTrain[, 1]),
                                       family = mboost::Binomial(), 
                                       control = mboost::boost_control(mstop = max(mstop_seq), nu = nu),
                                       center = FALSE)
      
      for (i in seq_along(mstop_seq)) {
        mseq <- mstop_seq[i]
        # Internal Validation
        pred_Boost_CV <- as.vector(stats::predict(fit_Boost_CV[mseq], newdata = as.matrix(XTest[, -1]), type = "response"))
        
        # Determine AUC using pROC package
        AUC_CV[i, j] <- pROC::auc(response = as.factor(XTest[, 1]), predictor = pred_Boost_CV, quiet = TRUE)
      }
    }
  }
  
  # Calculate the average AUC and choose the best mstop
  AUC <- rowMeans(AUC_CV, na.rm = TRUE)
  chosen_mstop <- mstop_seq[which.max(AUC)]
  
  # Train the final model with the chosen mstop
  final_model <- mboost::glmboost(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]),
                                  family = mboost::Binomial(), 
                                  control = mboost::boost_control(mstop = chosen_mstop, nu = nu),
                                  center = FALSE)
  
  # Return the result
  res <- list(
    best_mstop = chosen_mstop,
    best_model = final_model
  )
  
  # Set class
  class(res) <- "IntBoost"
  return(res)
}