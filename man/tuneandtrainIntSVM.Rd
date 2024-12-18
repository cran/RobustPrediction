% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tuneandtrainIntSVM.R
\name{tuneandtrainIntSVM}
\alias{tuneandtrainIntSVM}
\title{Tune and Train Internal SVM}
\usage{
tuneandtrainIntSVM(
  data,
  kernel = "linear",
  cost_seq = 2^(-15:15),
  scale = FALSE,
  nfolds = 5,
  seed = 123
)
}
\arguments{
\item{data}{A data frame containing the training data. The first column should be the response variable (factor), 
and the remaining columns should be the predictor variables.}

\item{kernel}{A character string specifying the kernel type to be used in the SVM. Default is "linear".}

\item{cost_seq}{A numeric vector of cost values to be evaluated. Default is `2^(-15:15)`.}

\item{scale}{A logical indicating whether to scale the predictor variables. Default is FALSE.}

\item{nfolds}{An integer specifying the number of folds for cross-validation. Default is 5.}

\item{seed}{An integer specifying the random seed for reproducibility. Default is 123.}
}
\value{
A list containing the best cost value (`best_cost`) and the final trained model (`best_model`).
}
\description{
This function tunes and trains a Support Vector Machine (SVM) classifier using the \code{mlr} package. 
The function evaluates a sequence of cost values using internal cross-validation and selects 
the best model based on the Area Under the Curve (AUC).
}
\details{
In Support Vector Machines, the \code{cost} parameter controls the trade-off between 
achieving a low training error and a low testing error. 
This function trains an SVM model on the training dataset, performs cross-validation, and 
selects the cost value that results in the highest AUC. The final model is then trained using the optimal 
cost value, and the performance is reported based on the AUC.
}
\examples{
\donttest{
# Load sample data
data(sample_data_train)

# Example usage
result <- tuneandtrainIntSVM(
  sample_data_train,
  kernel = "linear",
  cost_seq = 2^(-15:15),
  scale = FALSE,
  nfolds = 5,
  seed = 123
)
result$best_cost
result$best_model
}
}
