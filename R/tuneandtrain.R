#' Tune and Train Classifier
#'
#' This function tunes and trains a classifier using a specified tuning method. Depending on the method chosen, 
#' the function will either perform RobustTuneC, external tuning, or internal tuning.
#'
#' @param data A data frame containing the training data. The first column should be the response variable, 
#' which must be a factor for classification tasks. The remaining columns should be the predictor variables. 
#' Ensure that the data is properly formatted, with no missing values.
#' 
#' @param dataext A data frame containing the external validation data, required only for the tuning methods 
#' "robusttunec" and "ext". Similar to the `data` argument, the first column should be the response variable (factor), 
#' and the remaining columns should be the predictors. If `tuningmethod = "int"`, this parameter is ignored.
#'
#' @param tuningmethod A character string specifying which tuning approach to use. Options are:
#'   \itemize{
#'     \item "robusttunec": Uses robust tuning that combines internal and external validation for parameter selection.
#'     \item "ext": Uses external validation data for tuning the parameters.
#'     \item "int": Internal cross-validation is used to tune the parameters without any external data.
#'   }
#' 
#' @param classifier A character string specifying which classifier to use. Options include:
#'   \itemize{
#'     \item "boosting": Boosting algorithms for improving weak classifiers.
#'     \item "rf": Random Forest for robust decision tree-based models.
#'     \item "lasso": Lasso regression for feature selection and regularization.
#'     \item "ridge": Ridge regression for regularization.
#'     \item "svm": Support Vector Machines for high-dimensional classification.
#'   }
#'
#' @param ... Additional parameters to be passed to the specific tuning and training functions. These can include 
#' options such as the number of trees for Random Forest, the number of folds for cross-validation, or hyperparameters 
#' specific to the chosen classifier.
#'
#' @return A list containing the results of the tuning and training process, which typically includes:
#' \itemize{
#'   \item Best hyperparameters selected during the tuning process.
#'   \item The final trained model.
#'   \item Performance metrics (AUC) on the training or validation data, depending on the tuning method.
#' }
#'
#' @export
#'
#' @examples
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage: Robust tuning with Ridge classifier
#' result_boosting <- tuneandtrain(sample_data_train, sample_data_extern, 
#'   tuningmethod = "robusttunec", classifier = "ridge")
#' result_boosting$best_lambda
#' result_boosting$best_model
#' result_boosting$final_auc
#'
#' # Example usage: Internal cross-validation with Lasso classifier
#' result_lasso <- tuneandtrain(sample_data_train, tuningmethod = "int", 
#'   classifier = "lasso", maxit = 120000, nlambda = 200, nfolds = 5)
#' result_lasso$best_lambda
#' result_lasso$best_model
#' result_lasso$final_auc
#' result_lasso$active_set_Train

tuneandtrain <- function(data, dataext = NULL, tuningmethod, classifier, ...) {
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  
  # Initialize result
  res <- NULL
  
  # Choose the tuning method and call the respective function
  if (tuningmethod == "robusttunec") {
    if (is.null(dataext)) stop("dataext is required for the 'robusttunec' method.")
    dataext <- as.data.frame(dataext)
    res <- tuneandtrainRobustTuneC(data, dataext, classifier, ...)
  } else if (tuningmethod == "ext") {
    if (is.null(dataext)) stop("dataext is required for the 'ext' method.")
    dataext <- as.data.frame(dataext)
    res <- tuneandtrainExt(data, dataext, classifier, ...) 
  } else if (tuningmethod == "int") {
    res <- tuneandtrainInt(data, classifier, ...)
  } else {
    stop("Unknown tuning method specified.")
  }
  
  return(res)
}