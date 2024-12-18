% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/package.R
\docType{package}
\name{RobustPrediction}
\alias{RobustPrediction}
\alias{RobustPrediction-package}
\title{Package Title: Robust Tuning and Training for Cross-Source Prediction}
\description{
This package provides robust parameter tuning and predictive modeling techniques, useful for situations 
where prediction across different data sources is important and the data distribution varies slightly from source to source.
}
\details{
The 'RobustPrediction' package helps users build and tune classifiers using the methods  
'RobustTuneC' method, internal, or external tuning method. The package supports the following classifiers: 
boosting, lasso, ridge, random forest, and support vector machine(SVM). It is intended for scenarios 
where parameter tuning across data sources is important.


The 'RobustPrediction' package provides comprehensive tools for robust parameter tuning 
and predictive modeling, particularly for cross-source prediction tasks. 

The package includes functions for tuning model parameters using three methods:
- **Internal tuning**: Standard cross-validation on the training data to select the best parameters.
- **External tuning**: Parameter tuning based on an external dataset that is independent of the training data. This method 
  has two variants controlled by the \code{estperf} argument:
  - **Standard external tuning (\code{estperf = FALSE})**: Parameters are tuned directly using the external dataset. 
    This is the default approach and provides a straightforward method for selecting optimal parameters based on external data.
  - **Conservative external tuning (\code{estperf = TRUE})**: Internal tuning is first performed on the training data, 
    and then the model is evaluated on the external dataset. This approach provides a more conservative (slightly pessimistic) 
    AUC estimate, as described by Ellenbach et al. (2021). For the most accurate performance evaluation, 
    it is recommended to use a second external dataset.
- **RobustTuneC**: A method designed to combine internal and external tuning for better performance in cross-source scenarios.

The package supports Lasso, Ridge, Random Forest, Boosting, and SVM classifiers. 
These models can be trained and tuned using the provided methods, and the package includes 
the model's AUC (Area Under the Curve) value to help users evaluate prediction performance.

It is particularly useful when the data to be predicted comes from a different source than the training data, 
where variability between datasets may require more robust parameter tuning techniques. The methods provided in 
this package may help reduce overfitting the training data distribution and improve model generalization across 
different data sources.
}
\section{Dependencies}{

This package requires the following packages: \code{glmnet}, \code{mboost}, \code{mlr}, 
\code{pROC}, \code{ranger}.
}

\examples{
# Example usage:
data(sample_data_train)
data(sample_data_extern)
res <- tuneandtrain(sample_data_train, sample_data_extern, tuningmethod = "robusttunec", 
  classifier = "lasso")

}
\references{
Ellenbach, N., Boulesteix, A.-L., Bischl, B., Unger, K., & Hornung, R. (2021). 
Improved outcome prediction across data sources through robust parameter tuning. 
\emph{Journal of Classification}, \emph{38}, 212-231. 
<doi:10.1007/s00357-020-09368-z>.
}
\seealso{
Useful links:
\itemize{
  \item \url{https://github.com/Yuting-He/RobustPrediction}
}

}
\author{
\strong{Maintainer}: Yuting He \email{yutingh19@gmail.com}

Other contributors:
\itemize{
  \item Nicole Ellenbach [contributor]
  \item Roman Hornung [contributor]
}

}
