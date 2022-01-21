\name{subsemble-package}
\alias{subsemble-package}
\alias{subsemble-package}
\docType{package}
\title{
An Ensemble Method for Combining Subset-Specific Algorithm Fits
}
\description{
The Subsemble algorithm is a general subset ensemble prediction method, which can be used for small, moderate, or large datasets. Subsemble partitions the full dataset into subsets of observations, fits a specified underlying algorithm on each subset, and uses a unique form of k-fold cross-validation to output a prediction function that combines the subset-specific fits. An oracle result provides a theoretical performance guarantee for Subsemble.
}
\details{
\tabular{ll}{
Package: \tab subsemble\cr
Type: \tab Package\cr
Version: \tab 0.1.0\cr
Date: \tab 2012-01-22\cr
License: \tab Apache License (== 2.0)\cr
}
}
\author{
Authors: Erin LeDell, Stephanie Sapp, Mark van der Laan

Maintainer: Erin LeDell <oss@ledell.org>
}
\references{
LeDell, E. (2015) Scalable Ensemble Learning and Computationally Efficient Variance Estimation (Doctoral Dissertation).  University of California, Berkeley, USA.\cr
\url{https://github.com/ledell/phd-thesis/blob/main/ledell-phd-thesis.pdf}\cr
\cr
Stephanie Sapp, Mark J. van der Laan & John Canny. (2014) Subsemble: An ensemble method for combining subset-specific algorithm fits.  Journal of Applied Statistics, 41(6):1247-1259.\cr
\url{https://www.tandfonline.com/doi/abs/10.1080/02664763.2013.864263}\cr
\url{https://biostats.bepress.com/ucbbiostat/paper313/}
}

\note{
This work was supported in part by the Doris Duke Charitable Foundation Grant No. 2011042.
}

\keyword{models}
\seealso{
\code{\link[SuperLearner:SuperLearner]{SuperLearner}}
}
