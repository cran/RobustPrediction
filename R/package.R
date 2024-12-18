#' Package Title: Robust Tuning and Training for Cross-Source Prediction
#'
#' This package provides robust parameter tuning and predictive modeling techniques, useful for situations 
#' where prediction across different data sources is important and the data distribution varies slightly from source to source.
#'
#' The 'RobustPrediction' package helps users build and tune classifiers using the methods  
#' 'RobustTuneC' method, internal, or external tuning method. The package supports the following classifiers: 
#' boosting, lasso, ridge, random forest, and support vector machine(SVM). It is intended for scenarios 
#' where parameter tuning across data sources is important.
#'
#' @docType package
#' @name RobustPrediction
#' @aliases RobustPrediction-package
#' @details
#' The 'RobustPrediction' package provides comprehensive tools for robust parameter tuning 
#' and predictive modeling, particularly for cross-source prediction tasks. 
#' 
#' The package includes functions for tuning model parameters using three methods:
#' - **Internal tuning**: Standard cross-validation on the training data to select the best parameters.
#' - **External tuning**: Parameter tuning based on an external dataset that is independent of the training data. This method 
#'   has two variants controlled by the \code{estperf} argument:
#'   - **Standard external tuning (\code{estperf = FALSE})**: Parameters are tuned directly using the external dataset. 
#'     This is the default approach and provides a straightforward method for selecting optimal parameters based on external data.
#'   - **Conservative external tuning (\code{estperf = TRUE})**: Internal tuning is first performed on the training data, 
#'     and then the model is evaluated on the external dataset. This approach provides a more conservative (slightly pessimistic) 
#'     AUC estimate, as described by Ellenbach et al. (2021). For the most accurate performance evaluation, 
#'     it is recommended to use a second external dataset.
#' - **RobustTuneC**: A method designed to combine internal and external tuning for better performance in cross-source scenarios.
#' 
#' The package supports Lasso, Ridge, Random Forest, Boosting, and SVM classifiers. 
#' These models can be trained and tuned using the provided methods, and the package includes 
#' the model's AUC (Area Under the Curve) value to help users evaluate prediction performance.
#' 
#' It is particularly useful when the data to be predicted comes from a different source than the training data, 
#' where variability between datasets may require more robust parameter tuning techniques. The methods provided in 
#' this package may help reduce overfitting the training data distribution and improve model generalization across 
#' different data sources.
#'
#' @section Dependencies:
#' This package requires the following packages: \code{glmnet}, \code{mboost}, \code{mlr}, 
#' \code{pROC}, \code{ranger}.
#'
#' @examples
#' # Example usage:
#' data(sample_data_train)
#' data(sample_data_extern)
#' res <- tuneandtrain(sample_data_train, sample_data_extern, tuningmethod = "robusttunec", 
#'   classifier = "lasso")
#'
#' @references
#' Ellenbach, N., Boulesteix, A.-L., Bischl, B., Unger, K., & Hornung, R. (2021). 
#' Improved outcome prediction across data sources through robust parameter tuning. 
#' \emph{Journal of Classification}, \emph{38}, 212-231. 
#' <doi:10.1007/s00357-020-09368-z>.
"_PACKAGE"

#' Sample Training Data Subset
#'
#' This dataset, named `sample_data_train`, is a subset of publicly available microarray data from the HG-U133PLUS2 chip. 
#' It contains expression levels of 200 genes across 50 samples, used primarily as a training set in robust 
#' feature selection studies. 
#' The data has been sourced from the ArrayExpress repository and has been referenced in several research articles.
#'
#' @format A data frame with 50 observations and 201 variables, including:
#' \describe{
#'   \item{y}{Factor. The response variable.}
#'   \item{236694_at}{Numeric. Expression level of gene 236694_at.}
#'   \item{222356_at}{Numeric. Expression level of gene 222356_at.}
#'   \item{1554125_a_at}{Numeric. Expression level of gene 1554125_a_at.}
#'   \item{232823_at}{Numeric. Expression level of gene 232823_at.}
#'   \item{205766_at}{Numeric. Expression level of gene 205766_at.}
#'   \item{1560446_at}{Numeric. Expression level of gene 1560446_at.}
#'   \item{202565_s_at}{Numeric. Expression level of gene 202565_s_at.}
#'   \item{234887_at}{Numeric. Expression level of gene 234887_at.}
#'   \item{209687_at}{Numeric. Expression level of gene 209687_at.}
#'   \item{221592_at}{Numeric. Expression level of gene 221592_at.}
#'   \item{1570123_at}{Numeric. Expression level of gene 1570123_at.}
#'   \item{241368_at}{Numeric. Expression level of gene 241368_at.}
#'   \item{243324_x_at}{Numeric. Expression level of gene 243324_x_at.}
#'   \item{224046_s_at}{Numeric. Expression level of gene 224046_s_at.}
#'   \item{202775_s_at}{Numeric. Expression level of gene 202775_s_at.}
#'   \item{216332_at}{Numeric. Expression level of gene 216332_at.}
#'   \item{1569545_at}{Numeric. Expression level of gene 1569545_at.}
#'   \item{205946_at}{Numeric. Expression level of gene 205946_at.}
#'   \item{203547_at}{Numeric. Expression level of gene 203547_at.}
#'   \item{243239_at}{Numeric. Expression level of gene 243239_at.}
#'   \item{234245_at}{Numeric. Expression level of gene 234245_at.}
#'   \item{210832_x_at}{Numeric. Expression level of gene 210832_x_at.}
#'   \item{224549_x_at}{Numeric. Expression level of gene 224549_x_at.}
#'   \item{236628_at}{Numeric. Expression level of gene 236628_at.}
#'   \item{214848_at}{Numeric. Expression level of gene 214848_at.}
#'   \item{1553015_a_at}{Numeric. Expression level of gene 1553015_a_at.}
#'   \item{1554199_at}{Numeric. Expression level of gene 1554199_at.}
#'   \item{1557636_a_at}{Numeric. Expression level of gene 1557636_a_at.}
#'   \item{1558511_s_at}{Numeric. Expression level of gene 1558511_s_at.}
#'   \item{1561713_at}{Numeric. Expression level of gene 1561713_at.}
#'   \item{1561883_at}{Numeric. Expression level of gene 1561883_at.}
#'   \item{1568720_at}{Numeric. Expression level of gene 1568720_at.}
#'   \item{1569168_at}{Numeric. Expression level of gene 1569168_at.}
#'   \item{1569443_s_at}{Numeric. Expression level of gene 1569443_s_at.}
#'   \item{1570103_at}{Numeric. Expression level of gene 1570103_at.}
#'   \item{200916_at}{Numeric. Expression level of gene 200916_at.}
#'   \item{201554_x_at}{Numeric. Expression level of gene 201554_x_at.}
#'   \item{202371_at}{Numeric. Expression level of gene 202371_at.}
#'   \item{204481_at}{Numeric. Expression level of gene 204481_at.}
#'   \item{205831_at}{Numeric. Expression level of gene 205831_at.}
#'   \item{207061_at}{Numeric. Expression level of gene 207061_at.}
#'   \item{207423_s_at}{Numeric. Expression level of gene 207423_s_at.}
#'   \item{209896_s_at}{Numeric. Expression level of gene 209896_s_at.}
#'   \item{212646_at}{Numeric. Expression level of gene 212646_at.}
#'   \item{214068_at}{Numeric. Expression level of gene 214068_at.}
#'   \item{217727_x_at}{Numeric. Expression level of gene 217727_x_at.}
#'   \item{221103_s_at}{Numeric. Expression level of gene 221103_s_at.}
#'   \item{221785_at}{Numeric. Expression level of gene 221785_at.}
#'   \item{224207_x_at}{Numeric. Expression level of gene 224207_x_at.}
#'   \item{228257_at}{Numeric. Expression level of gene 228257_at.}
#'   \item{228877_at}{Numeric. Expression level of gene 228877_at.}
#'   \item{231173_at}{Numeric. Expression level of gene 231173_at.}
#'   \item{231328_s_at}{Numeric. Expression level of gene 231328_s_at.}
#'   \item{231639_at}{Numeric. Expression level of gene 231639_at.}
#'   \item{232221_x_at}{Numeric. Expression level of gene 232221_x_at.}
#'   \item{232349_x_at}{Numeric. Expression level of gene 232349_x_at.}
#'   \item{232849_at}{Numeric. Expression level of gene 232849_at.}
#'   \item{233601_at}{Numeric. Expression level of gene 233601_at.}
#'   \item{234403_at}{Numeric. Expression level of gene 234403_at.}
#'   \item{234585_at}{Numeric. Expression level of gene 234585_at.}
#'   \item{234650_at}{Numeric. Expression level of gene 234650_at.}
#'   \item{234897_s_at}{Numeric. Expression level of gene 234897_s_at.}
#'   \item{236071_at}{Numeric. Expression level of gene 236071_at.}
#'   \item{236689_at}{Numeric. Expression level of gene 236689_at.}
#'   \item{238551_at}{Numeric. Expression level of gene 238551_at.}
#'   \item{239414_at}{Numeric. Expression level of gene 239414_at.}
#'   \item{241034_at}{Numeric. Expression level of gene 241034_at.}
#'   \item{241131_at}{Numeric. Expression level of gene 241131_at.}
#'   \item{241897_at}{Numeric. Expression level of gene 241897_at.}
#'   \item{242611_at}{Numeric. Expression level of gene 242611_at.}
#'   \item{244805_at}{Numeric. Expression level of gene 244805_at.}
#'   \item{244866_at}{Numeric. Expression level of gene 244866_at.}
#'   \item{32259_at}{Numeric. Expression level of gene 32259_at.}
#'   \item{1552264_a_at}{Numeric. Expression level of gene 1552264_a_at.}
#'   \item{1552880_at}{Numeric. Expression level of gene 1552880_at.}
#'   \item{1553186_x_at}{Numeric. Expression level of gene 1553186_x_at.}
#'   \item{1553372_at}{Numeric. Expression level of gene 1553372_at.}
#'   \item{1553438_at}{Numeric. Expression level of gene 1553438_at.}
#'   \item{1554299_at}{Numeric. Expression level of gene 1554299_at.}
#'   \item{1554362_at}{Numeric. Expression level of gene 1554362_at.}
#'   \item{1554491_a_at}{Numeric. Expression level of gene 1554491_a_at.}
#'   \item{1555098_a_at}{Numeric. Expression level of gene 1555098_a_at.}
#'   \item{1555990_at}{Numeric. Expression level of gene 1555990_at.}
#'   \item{1556034_s_at}{Numeric. Expression level of gene 1556034_s_at.}
#'   \item{1556822_s_at}{Numeric. Expression level of gene 1556822_s_at.}
#'   \item{1556824_at}{Numeric. Expression level of gene 1556824_at.}
#'   \item{1557278_s_at}{Numeric. Expression level of gene 1557278_s_at.}
#'   \item{1558603_at}{Numeric. Expression level of gene 1558603_at.}
#'   \item{1558890_at}{Numeric. Expression level of gene 1558890_at.}
#'   \item{1560791_at}{Numeric. Expression level of gene 1560791_at.}
#'   \item{1561083_at}{Numeric. Expression level of gene 1561083_at.}
#'   \item{1561364_at}{Numeric. Expression level of gene 1561364_at.}
#'   \item{1561553_at}{Numeric. Expression level of gene 1561553_at.}
#'   \item{1562523_at}{Numeric. Expression level of gene 1562523_at.}
#'   \item{1562613_at}{Numeric. Expression level of gene 1562613_at.}
#'   \item{1563351_at}{Numeric. Expression level of gene 1563351_at.}
#'   \item{1563473_at}{Numeric. Expression level of gene 1563473_at.}
#'   \item{1566780_at}{Numeric. Expression level of gene 1566780_at.}
#'   \item{1567257_at}{Numeric. Expression level of gene 1567257_at.}
#'   \item{1569664_at}{Numeric. Expression level of gene 1569664_at.}
#'   \item{1569882_at}{Numeric. Expression level of gene 1569882_at.}
#'   \item{1570252_at}{Numeric. Expression level of gene 1570252_at.}
#'   \item{201089_at}{Numeric. Expression level of gene 201089_at.}
#'   \item{201261_x_at}{Numeric. Expression level of gene 201261_x_at.}
#'   \item{202052_s_at}{Numeric. Expression level of gene 202052_s_at.}
#'   \item{202236_s_at}{Numeric. Expression level of gene 202236_s_at.}
#'   \item{202948_at}{Numeric. Expression level of gene 202948_at.}
#'   \item{203080_s_at}{Numeric. Expression level of gene 203080_s_at.}
#'   \item{203211_s_at}{Numeric. Expression level of gene 203211_s_at.}
#'   \item{203218_at}{Numeric. Expression level of gene 203218_at.}
#'   \item{203236_s_at}{Numeric. Expression level of gene 203236_s_at.}
#'   \item{203347_s_at}{Numeric. Expression level of gene 203347_s_at.}
#'   \item{203960_s_at}{Numeric. Expression level of gene 203960_s_at.}
#'   \item{204609_at}{Numeric. Expression level of gene 204609_at.}
#'   \item{204806_x_at}{Numeric. Expression level of gene 204806_x_at.}
#'   \item{204949_at}{Numeric. Expression level of gene 204949_at.}
#'   \item{204979_s_at}{Numeric. Expression level of gene 204979_s_at.}
#'   \item{205823_at}{Numeric. Expression level of gene 205823_at.}
#'   \item{205902_at}{Numeric. Expression level of gene 205902_at.}
#'   \item{205967_at}{Numeric. Expression level of gene 205967_at.}
#'   \item{206186_at}{Numeric. Expression level of gene 206186_at.}
#'   \item{207151_at}{Numeric. Expression level of gene 207151_at.}
#'   \item{207379_at}{Numeric. Expression level of gene 207379_at.}
#'   \item{207440_at}{Numeric. Expression level of gene 207440_at.}
#'   \item{207883_s_at}{Numeric. Expression level of gene 207883_s_at.}
#'   \item{208277_at}{Numeric. Expression level of gene 208277_at.}
#'   \item{208280_at}{Numeric. Expression level of gene 208280_at.}
#'   \item{209224_s_at}{Numeric. Expression level of gene 209224_s_at.}
#'   \item{209561_at}{Numeric. Expression level of gene 209561_at.}
#'   \item{209630_s_at}{Numeric. Expression level of gene 209630_s_at.}
#'   \item{210118_s_at}{Numeric. Expression level of gene 210118_s_at.}
#'   \item{210342_s_at}{Numeric. Expression level of gene 210342_s_at.}
#'   \item{211566_x_at}{Numeric. Expression level of gene 211566_x_at.}
#'   \item{211756_at}{Numeric. Expression level of gene 211756_at.}
#'   \item{212170_at}{Numeric. Expression level of gene 212170_at.}
#'   \item{212494_at}{Numeric. Expression level of gene 212494_at.}
#'   \item{213118_at}{Numeric. Expression level of gene 213118_at.}
#'   \item{214475_x_at}{Numeric. Expression level of gene 214475_x_at.}
#'   \item{214834_at}{Numeric. Expression level of gene 214834_at.}
#'   \item{215718_s_at}{Numeric. Expression level of gene 215718_s_at.}
#'   \item{216283_s_at}{Numeric. Expression level of gene 216283_s_at.}
#'   \item{217206_at}{Numeric. Expression level of gene 217206_at.}
#'   \item{217557_s_at}{Numeric. Expression level of gene 217557_s_at.}
#'   \item{217577_at}{Numeric. Expression level of gene 217577_at.}
#'   \item{218152_at}{Numeric. Expression level of gene 218152_at.}
#'   \item{218252_at}{Numeric. Expression level of gene 218252_at.}
#'   \item{219714_s_at}{Numeric. Expression level of gene 219714_s_at.}
#'   \item{220506_at}{Numeric. Expression level of gene 220506_at.}
#'   \item{220889_s_at}{Numeric. Expression level of gene 220889_s_at.}
#'   \item{221204_s_at}{Numeric. Expression level of gene 221204_s_at.}
#'   \item{221795_at}{Numeric. Expression level of gene 221795_at.}
#'   \item{222048_at}{Numeric. Expression level of gene 222048_at.}
#'   \item{223142_s_at}{Numeric. Expression level of gene 223142_s_at.}
#'   \item{223439_at}{Numeric. Expression level of gene 223439_at.}
#'   \item{223673_at}{Numeric. Expression level of gene 223673_at.}
#'   \item{224363_at}{Numeric. Expression level of gene 224363_at.}
#'   \item{224512_s_at}{Numeric. Expression level of gene 224512_s_at.}
#'   \item{224690_at}{Numeric. Expression level of gene 224690_at.}
#'   \item{224936_at}{Numeric. Expression level of gene 224936_at.}
#'   \item{225334_at}{Numeric. Expression level of gene 225334_at.}
#'   \item{225713_at}{Numeric. Expression level of gene 225713_at.}
#'   \item{225839_at}{Numeric. Expression level of gene 225839_at.}
#'   \item{226041_at}{Numeric. Expression level of gene 226041_at.}
#'   \item{226093_at}{Numeric. Expression level of gene 226093_at.}
#'   \item{226543_at}{Numeric. Expression level of gene 226543_at.}
#'   \item{227695_at}{Numeric. Expression level of gene 227695_at.}
#'   \item{228295_at}{Numeric. Expression level of gene 228295_at.}
#'   \item{228548_at}{Numeric. Expression level of gene 228548_at.}
#'   \item{229234_at}{Numeric. Expression level of gene 229234_at.}
#'   \item{229658_at}{Numeric. Expression level of gene 229658_at.}
#'   \item{229725_at}{Numeric. Expression level of gene 229725_at.}
#'   \item{230252_at}{Numeric. Expression level of gene 230252_at.}
#'   \item{230471_at}{Numeric. Expression level of gene 230471_at.}
#'   \item{231149_s_at}{Numeric. Expression level of gene 231149_s_at.}
#'   \item{231556_at}{Numeric. Expression level of gene 231556_at.}
#'   \item{231754_at}{Numeric. Expression level of gene 231754_at.}
#'   \item{232011_s_at}{Numeric. Expression level of gene 232011_s_at.}
#'   \item{233030_at}{Numeric. Expression level of gene 233030_at.}
#'   \item{234161_at}{Numeric. Expression level of gene 234161_at.}
#'   \item{235050_at}{Numeric. Expression level of gene 235050_at.}
#'   \item{235094_at}{Numeric. Expression level of gene 235094_at.}
#'   \item{235278_at}{Numeric. Expression level of gene 235278_at.}
#'   \item{235671_at}{Numeric. Expression level of gene 235671_at.}
#'   \item{235952_at}{Numeric. Expression level of gene 235952_at.}
#'   \item{236158_at}{Numeric. Expression level of gene 236158_at.}
#'   \item{236181_at}{Numeric. Expression level of gene 236181_at.}
#'   \item{237055_at}{Numeric. Expression level of gene 237055_at.}
#'   \item{237768_x_at}{Numeric. Expression level of gene 237768_x_at.}
#'   \item{238897_at}{Numeric. Expression level of gene 238897_at.}
#'   \item{239160_at}{Numeric. Expression level of gene 239160_at.}
#'   \item{239998_at}{Numeric. Expression level of gene 239998_at.}
#'   \item{240254_at}{Numeric. Expression level of gene 240254_at.}
#'   \item{240612_at}{Numeric. Expression level of gene 240612_at.}
#'   \item{240692_at}{Numeric. Expression level of gene 240692_at.}
#'   \item{240822_at}{Numeric. Expression level of gene 240822_at.}
#'   \item{240842_at}{Numeric. Expression level of gene 240842_at.}
#'   \item{241331_at}{Numeric. Expression level of gene 241331_at.}
#'   \item{241598_at}{Numeric. Expression level of gene 241598_at.}
#'   \item{241927_x_at}{Numeric. Expression level of gene 241927_x_at.}
#'   \item{242405_at}{Numeric. Expression level of gene 242405_at.}
#' }
#'
#' @details 
#' This dataset was extracted from a larger dataset available on ArrayExpress. It is used as a training set 
#' for feature selection tasks and other machine learning applications in bioinformatics.
#'
#' @source 
#' The original dataset can be found on ArrayExpress: \url{https://www.ebi.ac.uk/arrayexpress}
#'
#' @references
#' Ellenbach, N., Boulesteix, A.L., Bischl, B., et al. (2021). 
#' Improved Outcome Prediction Across Data Sources Through Robust Parameter Tuning. \emph{Journal of Classification}, 
#' 38, 212–231. \doi{10.1007/s00357-020-09368-z}.
#' 
#' Hornung, R., Causeur, D., Bernau, C., Boulesteix, A.L. (2017). 
#' Improving cross-study prediction through addon batch effect adjustment or addon normalization. \emph{Bioinformatics}, 
#' 33(3), 397–404. \doi{10.1093/bioinformatics/btw650}.
#'
#' @examples
#' # Load the dataset:
#' data(sample_data_train)
#' 
#' # Dimension of the dataset:
#' dim(sample_data_train)
#' 
#' # View the first rows of the dataset:
#' head(sample_data_train)
"sample_data_train"

#' Sample External Validation Data Subset
#'
#' This dataset, named `sample_data_extern`, is a subset of publicly available microarray data from the HG-U133PLUS2 chip. 
#' It contains expression levels of 200 genes across 50 samples, used primarily as an external validation set 
#' in robust feature selection studies. 
#' The data has been sourced from the ArrayExpress repository and has been referenced in several research articles.
#'
#' @format A data frame with 50 observations and 201 variables, including:
#' \describe{
#'   \item{y}{Factor. The response variable.}
#'   \item{236694_at}{Numeric. Expression level of gene 236694_at.}
#'   \item{222356_at}{Numeric. Expression level of gene 222356_at.}
#'   \item{1554125_a_at}{Numeric. Expression level of gene 1554125_a_at.}
#'   \item{232823_at}{Numeric. Expression level of gene 232823_at.}
#'   \item{205766_at}{Numeric. Expression level of gene 205766_at.}
#'   \item{1560446_at}{Numeric. Expression level of gene 1560446_at.}
#'   \item{202565_s_at}{Numeric. Expression level of gene 202565_s_at.}
#'   \item{234887_at}{Numeric. Expression level of gene 234887_at.}
#'   \item{209687_at}{Numeric. Expression level of gene 209687_at.}
#'   \item{221592_at}{Numeric. Expression level of gene 221592_at.}
#'   \item{1570123_at}{Numeric. Expression level of gene 1570123_at.}
#'   \item{241368_at}{Numeric. Expression level of gene 241368_at.}
#'   \item{243324_x_at}{Numeric. Expression level of gene 243324_x_at.}
#'   \item{224046_s_at}{Numeric. Expression level of gene 224046_s_at.}
#'   \item{202775_s_at}{Numeric. Expression level of gene 202775_s_at.}
#'   \item{216332_at}{Numeric. Expression level of gene 216332_at.}
#'   \item{1569545_at}{Numeric. Expression level of gene 1569545_at.}
#'   \item{205946_at}{Numeric. Expression level of gene 205946_at.}
#'   \item{203547_at}{Numeric. Expression level of gene 203547_at.}
#'   \item{243239_at}{Numeric. Expression level of gene 243239_at.}
#'   \item{234245_at}{Numeric. Expression level of gene 234245_at.}
#'   \item{210832_x_at}{Numeric. Expression level of gene 210832_x_at.}
#'   \item{224549_x_at}{Numeric. Expression level of gene 224549_x_at.}
#'   \item{236628_at}{Numeric. Expression level of gene 236628_at.}
#'   \item{214848_at}{Numeric. Expression level of gene 214848_at.}
#'   \item{1553015_a_at}{Numeric. Expression level of gene 1553015_a_at.}
#'   \item{1554199_at}{Numeric. Expression level of gene 1554199_at.}
#'   \item{1557636_a_at}{Numeric. Expression level of gene 1557636_a_at.}
#'   \item{1558511_s_at}{Numeric. Expression level of gene 1558511_s_at.}
#'   \item{1561713_at}{Numeric. Expression level of gene 1561713_at.}
#'   \item{1561883_at}{Numeric. Expression level of gene 1561883_at.}
#'   \item{1568720_at}{Numeric. Expression level of gene 1568720_at.}
#'   \item{1569168_at}{Numeric. Expression level of gene 1569168_at.}
#'   \item{1569443_s_at}{Numeric. Expression level of gene 1569443_s_at.}
#'   \item{1570103_at}{Numeric. Expression level of gene 1570103_at.}
#'   \item{200916_at}{Numeric. Expression level of gene 200916_at.}
#'   \item{201554_x_at}{Numeric. Expression level of gene 201554_x_at.}
#'   \item{202371_at}{Numeric. Expression level of gene 202371_at.}
#'   \item{204481_at}{Numeric. Expression level of gene 204481_at.}
#'   \item{205831_at}{Numeric. Expression level of gene 205831_at.}
#'   \item{207061_at}{Numeric. Expression level of gene 207061_at.}
#'   \item{207423_s_at}{Numeric. Expression level of gene 207423_s_at.}
#'   \item{209896_s_at}{Numeric. Expression level of gene 209896_s_at.}
#'   \item{212646_at}{Numeric. Expression level of gene 212646_at.}
#'   \item{214068_at}{Numeric. Expression level of gene 214068_at.}
#'   \item{217727_x_at}{Numeric. Expression level of gene 217727_x_at.}
#'   \item{221103_s_at}{Numeric. Expression level of gene 221103_s_at.}
#'   \item{221785_at}{Numeric. Expression level of gene 221785_at.}
#'   \item{224207_x_at}{Numeric. Expression level of gene 224207_x_at.}
#'   \item{228257_at}{Numeric. Expression level of gene 228257_at.}
#'   \item{228877_at}{Numeric. Expression level of gene 228877_at.}
#'   \item{231173_at}{Numeric. Expression level of gene 231173_at.}
#'   \item{231328_s_at}{Numeric. Expression level of gene 231328_s_at.}
#'   \item{231639_at}{Numeric. Expression level of gene 231639_at.}
#'   \item{232221_x_at}{Numeric. Expression level of gene 232221_x_at.}
#'   \item{232349_x_at}{Numeric. Expression level of gene 232349_x_at.}
#'   \item{232849_at}{Numeric. Expression level of gene 232849_at.}
#'   \item{233601_at}{Numeric. Expression level of gene 233601_at.}
#'   \item{234403_at}{Numeric. Expression level of gene 234403_at.}
#'   \item{234585_at}{Numeric. Expression level of gene 234585_at.}
#'   \item{234650_at}{Numeric. Expression level of gene 234650_at.}
#'   \item{234897_s_at}{Numeric. Expression level of gene 234897_s_at.}
#'   \item{236071_at}{Numeric. Expression level of gene 236071_at.}
#'   \item{236689_at}{Numeric. Expression level of gene 236689_at.}
#'   \item{238551_at}{Numeric. Expression level of gene 238551_at.}
#'   \item{239414_at}{Numeric. Expression level of gene 239414_at.}
#'   \item{241034_at}{Numeric. Expression level of gene 241034_at.}
#'   \item{241131_at}{Numeric. Expression level of gene 241131_at.}
#'   \item{241897_at}{Numeric. Expression level of gene 241897_at.}
#'   \item{242611_at}{Numeric. Expression level of gene 242611_at.}
#'   \item{244805_at}{Numeric. Expression level of gene 244805_at.}
#'   \item{244866_at}{Numeric. Expression level of gene 244866_at.}
#'   \item{32259_at}{Numeric. Expression level of gene 32259_at.}
#'   \item{1552264_a_at}{Numeric. Expression level of gene 1552264_a_at.}
#'   \item{1552880_at}{Numeric. Expression level of gene 1552880_at.}
#'   \item{1553186_x_at}{Numeric. Expression level of gene 1553186_x_at.}
#'   \item{1553372_at}{Numeric. Expression level of gene 1553372_at.}
#'   \item{1553438_at}{Numeric. Expression level of gene 1553438_at.}
#'   \item{1554299_at}{Numeric. Expression level of gene 1554299_at.}
#'   \item{1554362_at}{Numeric. Expression level of gene 1554362_at.}
#'   \item{1554491_a_at}{Numeric. Expression level of gene 1554491_a_at.}
#'   \item{1555098_a_at}{Numeric. Expression level of gene 1555098_a_at.}
#'   \item{1555990_at}{Numeric. Expression level of gene 1555990_at.}
#'   \item{1556034_s_at}{Numeric. Expression level of gene 1556034_s_at.}
#'   \item{1556822_s_at}{Numeric. Expression level of gene 1556822_s_at.}
#'   \item{1556824_at}{Numeric. Expression level of gene 1556824_at.}
#'   \item{1557278_s_at}{Numeric. Expression level of gene 1557278_s_at.}
#'   \item{1558603_at}{Numeric. Expression level of gene 1558603_at.}
#'   \item{1558890_at}{Numeric. Expression level of gene 1558890_at.}
#'   \item{1560791_at}{Numeric. Expression level of gene 1560791_at.}
#'   \item{1561083_at}{Numeric. Expression level of gene 1561083_at.}
#'   \item{1561364_at}{Numeric. Expression level of gene 1561364_at.}
#'   \item{1561553_at}{Numeric. Expression level of gene 1561553_at.}
#'   \item{1562523_at}{Numeric. Expression level of gene 1562523_at.}
#'   \item{1562613_at}{Numeric. Expression level of gene 1562613_at.}
#'   \item{1563351_at}{Numeric. Expression level of gene 1563351_at.}
#'   \item{1563473_at}{Numeric. Expression level of gene 1563473_at.}
#'   \item{1566780_at}{Numeric. Expression level of gene 1566780_at.}
#'   \item{1567257_at}{Numeric. Expression level of gene 1567257_at.}
#'   \item{1569664_at}{Numeric. Expression level of gene 1569664_at.}
#'   \item{1569882_at}{Numeric. Expression level of gene 1569882_at.}
#'   \item{1570252_at}{Numeric. Expression level of gene 1570252_at.}
#'   \item{201089_at}{Numeric. Expression level of gene 201089_at.}
#'   \item{201261_x_at}{Numeric. Expression level of gene 201261_x_at.}
#'   \item{202052_s_at}{Numeric. Expression level of gene 202052_s_at.}
#'   \item{202236_s_at}{Numeric. Expression level of gene 202236_s_at.}
#'   \item{202948_at}{Numeric. Expression level of gene 202948_at.}
#'   \item{203080_s_at}{Numeric. Expression level of gene 203080_s_at.}
#'   \item{203211_s_at}{Numeric. Expression level of gene 203211_s_at.}
#'   \item{203218_at}{Numeric. Expression level of gene 203218_at.}
#'   \item{203236_s_at}{Numeric. Expression level of gene 203236_s_at.}
#'   \item{203347_s_at}{Numeric. Expression level of gene 203347_s_at.}
#'   \item{203960_s_at}{Numeric. Expression level of gene 203960_s_at.}
#'   \item{204609_at}{Numeric. Expression level of gene 204609_at.}
#'   \item{204806_x_at}{Numeric. Expression level of gene 204806_x_at.}
#'   \item{204949_at}{Numeric. Expression level of gene 204949_at.}
#'   \item{204979_s_at}{Numeric. Expression level of gene 204979_s_at.}
#'   \item{205823_at}{Numeric. Expression level of gene 205823_at.}
#'   \item{205902_at}{Numeric. Expression level of gene 205902_at.}
#'   \item{205967_at}{Numeric. Expression level of gene 205967_at.}
#'   \item{206186_at}{Numeric. Expression level of gene 206186_at.}
#'   \item{207151_at}{Numeric. Expression level of gene 207151_at.}
#'   \item{207379_at}{Numeric. Expression level of gene 207379_at.}
#'   \item{207440_at}{Numeric. Expression level of gene 207440_at.}
#'   \item{207883_s_at}{Numeric. Expression level of gene 207883_s_at.}
#'   \item{208277_at}{Numeric. Expression level of gene 208277_at.}
#'   \item{208280_at}{Numeric. Expression level of gene 208280_at.}
#'   \item{209224_s_at}{Numeric. Expression level of gene 209224_s_at.}
#'   \item{209561_at}{Numeric. Expression level of gene 209561_at.}
#'   \item{209630_s_at}{Numeric. Expression level of gene 209630_s_at.}
#'   \item{210118_s_at}{Numeric. Expression level of gene 210118_s_at.}
#'   \item{210342_s_at}{Numeric. Expression level of gene 210342_s_at.}
#'   \item{211566_x_at}{Numeric. Expression level of gene 211566_x_at.}
#'   \item{211756_at}{Numeric. Expression level of gene 211756_at.}
#'   \item{212170_at}{Numeric. Expression level of gene 212170_at.}
#'   \item{212494_at}{Numeric. Expression level of gene 212494_at.}
#'   \item{213118_at}{Numeric. Expression level of gene 213118_at.}
#'   \item{214475_x_at}{Numeric. Expression level of gene 214475_x_at.}
#'   \item{214834_at}{Numeric. Expression level of gene 214834_at.}
#'   \item{215718_s_at}{Numeric. Expression level of gene 215718_s_at.}
#'   \item{216283_s_at}{Numeric. Expression level of gene 216283_s_at.}
#'   \item{217206_at}{Numeric. Expression level of gene 217206_at.}
#'   \item{217557_s_at}{Numeric. Expression level of gene 217557_s_at.}
#'   \item{217577_at}{Numeric. Expression level of gene 217577_at.}
#'   \item{218152_at}{Numeric. Expression level of gene 218152_at.}
#'   \item{218252_at}{Numeric. Expression level of gene 218252_at.}
#'   \item{219714_s_at}{Numeric. Expression level of gene 219714_s_at.}
#'   \item{220506_at}{Numeric. Expression level of gene 220506_at.}
#'   \item{220889_s_at}{Numeric. Expression level of gene 220889_s_at.}
#'   \item{221204_s_at}{Numeric. Expression level of gene 221204_s_at.}
#'   \item{221795_at}{Numeric. Expression level of gene 221795_at.}
#'   \item{222048_at}{Numeric. Expression level of gene 222048_at.}
#'   \item{223142_s_at}{Numeric. Expression level of gene 223142_s_at.}
#'   \item{223439_at}{Numeric. Expression level of gene 223439_at.}
#'   \item{223673_at}{Numeric. Expression level of gene 223673_at.}
#'   \item{224363_at}{Numeric. Expression level of gene 224363_at.}
#'   \item{224512_s_at}{Numeric. Expression level of gene 224512_s_at.}
#'   \item{224690_at}{Numeric. Expression level of gene 224690_at.}
#'   \item{224936_at}{Numeric. Expression level of gene 224936_at.}
#'   \item{225334_at}{Numeric. Expression level of gene 225334_at.}
#'   \item{225713_at}{Numeric. Expression level of gene 225713_at.}
#'   \item{225839_at}{Numeric. Expression level of gene 225839_at.}
#'   \item{226041_at}{Numeric. Expression level of gene 226041_at.}
#'   \item{226093_at}{Numeric. Expression level of gene 226093_at.}
#'   \item{226543_at}{Numeric. Expression level of gene 226543_at.}
#'   \item{227695_at}{Numeric. Expression level of gene 227695_at.}
#'   \item{228295_at}{Numeric. Expression level of gene 228295_at.}
#'   \item{228548_at}{Numeric. Expression level of gene 228548_at.}
#'   \item{229234_at}{Numeric. Expression level of gene 229234_at.}
#'   \item{229658_at}{Numeric. Expression level of gene 229658_at.}
#'   \item{229725_at}{Numeric. Expression level of gene 229725_at.}
#'   \item{230252_at}{Numeric. Expression level of gene 230252_at.}
#'   \item{230471_at}{Numeric. Expression level of gene 230471_at.}
#'   \item{231149_s_at}{Numeric. Expression level of gene 231149_s_at.}
#'   \item{231556_at}{Numeric. Expression level of gene 231556_at.}
#'   \item{231754_at}{Numeric. Expression level of gene 231754_at.}
#'   \item{232011_s_at}{Numeric. Expression level of gene 232011_s_at.}
#'   \item{233030_at}{Numeric. Expression level of gene 233030_at.}
#'   \item{234161_at}{Numeric. Expression level of gene 234161_at.}
#'   \item{235050_at}{Numeric. Expression level of gene 235050_at.}
#'   \item{235094_at}{Numeric. Expression level of gene 235094_at.}
#'   \item{235278_at}{Numeric. Expression level of gene 235278_at.}
#'   \item{235671_at}{Numeric. Expression level of gene 235671_at.}
#'   \item{235952_at}{Numeric. Expression level of gene 235952_at.}
#'   \item{236158_at}{Numeric. Expression level of gene 236158_at.}
#'   \item{236181_at}{Numeric. Expression level of gene 236181_at.}
#'   \item{237055_at}{Numeric. Expression level of gene 237055_at.}
#'   \item{237768_x_at}{Numeric. Expression level of gene 237768_x_at.}
#'   \item{238897_at}{Numeric. Expression level of gene 238897_at.}
#'   \item{239160_at}{Numeric. Expression level of gene 239160_at.}
#'   \item{239998_at}{Numeric. Expression level of gene 239998_at.}
#'   \item{240254_at}{Numeric. Expression level of gene 240254_at.}
#'   \item{240612_at}{Numeric. Expression level of gene 240612_at.}
#'   \item{240692_at}{Numeric. Expression level of gene 240692_at.}
#'   \item{240822_at}{Numeric. Expression level of gene 240822_at.}
#'   \item{240842_at}{Numeric. Expression level of gene 240842_at.}
#'   \item{241331_at}{Numeric. Expression level of gene 241331_at.}
#'   \item{241598_at}{Numeric. Expression level of gene 241598_at.}
#'   \item{241927_x_at}{Numeric. Expression level of gene 241927_x_at.}
#'   \item{242405_at}{Numeric. Expression level of gene 242405_at.}
#' }
#'
#' @details 
#' This dataset was extracted from a larger dataset available on ArrayExpress and is used as an external validation set 
#' for feature selection tasks and other machine learning applications in bioinformatics.
#'
#' @source 
#' The original dataset can be found on ArrayExpress: \url{https://www.ebi.ac.uk/arrayexpress}
#'
#' @references
#' Ellenbach, N., Boulesteix, A.L., Bischl, B., et al. (2021). 
#' Improved Outcome Prediction Across Data Sources Through Robust Parameter Tuning. \emph{Journal of Classification}, 
#' 38, 212–231. \doi{10.1007/s00357-020-09368-z}.
#' 
#' Hornung, R., Causeur, D., Bernau, C., Boulesteix, A.L. (2017). 
#' Improving cross-study prediction through addon batch effect adjustment or addon normalization. \emph{Bioinformatics}, 
#' 33(3), 397–404. \doi{10.1093/bioinformatics/btw650}.
#'
#' @examples
#' # Load the dataset
#' data(sample_data_extern)
#' 
#' # View the first few rows of the dataset
#' head(sample_data_extern)
#' 
#' # Summary of the dataset
#' summary(sample_data_extern)
"sample_data_extern"
