% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tuneandtrainRobustTuneCLasso.R
\name{tuneandtrainRobustTuneCLasso}
\alias{tuneandtrainRobustTuneCLasso}
\title{Tune and Train RobustTuneC Lasso}
\usage{
tuneandtrainRobustTuneCLasso(
  data,
  dataext,
  K = 5,
  maxit = 120000,
  nlambda = 100
)
}
\arguments{
\item{data}{A data frame containing the training data. The first column should be the response variable (factor), 
and the remaining columns should be the predictor variables.}

\item{dataext}{A data frame containing the external validation data. The first column should be the response 
variable (factor), and the remaining columns should be the predictor variables.}

\item{K}{Number of folds to use in cross-validation. Default is 5.}

\item{maxit}{Maximum number of iterations. Default is 120000.}

\item{nlambda}{The number of lambda values to use for cross-validation. Default is 100.}
}
\value{
A list containing the best lambda value (`best_lambda`), the final trained model (`best_model`), 
  the number of active coefficients (`active_set_Train`), and the chosen c value(`best_c`).
}
\description{
This function tunes and trains a Lasso classifier using the \code{glmnet} package and the "RobustTuneC" method.
The function uses K-fold cross-validation to evaluate a sequence of lambda (regularization) values and selects 
the best model based on the Area Under the Curve (AUC).
}
\details{
This function trains a logistic Lasso model using the training dataset and validates it through cross-validation.
After selecting the best lambda value based on the training data, the model is then applied to an external validation dataset
to compute the final AUC. The lambda value that results in the highest AUC on the external validation dataset is chosen as the best model.
}
\examples{
# Load sample data
data(sample_data_train)
data(sample_data_extern)

# Example usage
result <- tuneandtrainRobustTuneCLasso(sample_data_train, sample_data_extern, 
  K = 5, maxit = 120000, nlambda = 100)
result$best_lambda
result$best_model
result$best_c
}
