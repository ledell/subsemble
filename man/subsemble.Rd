\name{subsemble}
\alias{subsemble}
\title{
An Ensemble Method for Combining Subset-Specific Algorithm Fits
}
\description{
Subsemble partitions the full dataset into subsets of observations, fits a specified underlying algorithm on each subset, and uses a unique form of k-fold cross-validation to output a prediction function that combines the subset-specific fits.
}
\usage{
subsemble(x, y, newx = NULL, family = gaussian(), 
    learner, metalearner = "SL.glm", subsets = 3, subControl = list(), 
    cvControl = list(), learnControl = list(), genControl = list(),
    id = NULL, obsWeights = NULL, seed = 1, parallel = "seq")
}

\arguments{
  \item{x}{
The data.frame or matrix of predictor variables.
}
  \item{y}{
The outcome in the training data set. Must be a numeric vector.
}
  \item{newx}{
The predictor variables in the test data set. The structure should match \code{x}. If missing, uses \code{x} for \code{newx}.
}
  \item{family}{
A description of the error distribution and link function to be used in the model. This can be a character string naming a family function, a family function or the result of a call to a family function.  (See '?family' for the  details of family functions.) Currently allows \code{gaussian()} or \code{binomial()}. 
}
  \item{learner}{
A string or character vector naming the prediction algorithm(s) used to train a model on each of the subsets of \code{x}.  This uses the learning algorithm API provided by the SuperLearner package, so for example, we could use \code{learner = "SL.randomForest"} or \code{learner = c("SL.glm","SL.randomForest")}.  See the \code{\link[SuperLearner:listWrappers]{listWrappers}} function for a full list of available algorithms.  If a single learner is provided, the same algorithm will be used on each of the subsets.  If a vector of learners is provided, then each algorithm will be applied to each of the subsets (default behavior specified by the \code{learnControl$multiType="crossprod"}); or alternatively, the multiple algorithms can be applied to different subsets with \code{learnControl$multiType="divisor"}.
}
  \item{metalearner}{
A string specifying the prediction algorithm used to learn the optimal weighted combination of the sublearners (ie. models learned on subsets of the data.)  This uses the API provided by the SuperLearner package, so for example, we could use \code{metalearner = "SL.glmnet"}.  See the \code{\link[SuperLearner:listWrappers]{listWrappers}} function for a full list of available algorithms.
}
  \item{subsets}{
An integer specifying the number of subsets the data should be partitioned into, a vector of subset labels equal to the number of rows of \code{x}, or a user-specified list of index vectors equal to the number of subsets.  If subsets is an integer, you can control how the subsets are partitioned (random shuffle, etc) using the \code{subControl} argument.
}
  \item{subControl}{
A list of parameters to control the data partitioning (subsetting) process. The logical \code{stratifyCV} list parameter will stratify the data splits by binary outcome (\code{family=binomial()} only), and defaults to \code{TRUE}.  The logical \code{shuffle} parameter defaults to \code{TRUE} to ensure that subsets will be created randomly.  If the user explicitly specifies the subsets via the \code{subsets} argument, that will override any parameters in this list.  The last parameter, \code{supervised}, currently defaults to \code{NULL} and is a place-holder for option to learn the optimal subsets in a supervised manner.  This will be implemented in a future release.
}
  \item{cvControl}{
A list of parameters to control the cross-validation process. The \code{V} parameter is an integer representing the number of cross-validation folds and defaults to 10.  Each of the subsets will be divided into \code{V} cross-validation folds.  The other parameters are \code{stratifyCV} and \code{shuffle}, which are both logical and default to \code{TRUE}.  See above for descriptions of these parameters.  
}
  \item{learnControl}{
A list of parameters to control the learning process.  Currently, the only parameter is \code{multiType}, which is only used if there are multiple learners specified by the \code{learner} argument.  The two supported values for \code{multiType} are \code{"crossprod"} (the default) and \code{"divisor"}.  The \code{"crossprod"} type will train each of the learners on each of the subsets.  For the \code{"divisor"} type, the length of the \code{learners} vector must be a divisor of the number of subsets.  If \code{length(learner)} equals the number of subsets, each learner will be applied to a single subset.  If \code{length(learner)} is a divisor of the number of subsets, then the learners will be repeated as necessary (to equal the number of subsets).
}
  \item{genControl}{
A list of general control parameters.  Currently, the only parameter is \code{saveFits}, which defaults to \code{TRUE}.  If set to \code{FALSE}, then the \code{subfits} and \code{metafit} output objects will be set to \code{NULL}.  This can be used if you want to train and test in one step and do not want to waste disk space storing all the models.
}
  \item{id}{
Optional cluster identification variable. Passed to the \code{learner} algorithm.
}
  \item{obsWeights}{
Optional observation weights vector. As with \code{id} above, \code{obsWeights} is passed to the prediction and screening algorithms, but many of the built in learner wrappers ignore (or can't use) the information. If you are using observation weights, make sure the learner you specify uses the information, or the weights will be ignored.
}
  \item{seed}{
A random seed to be set (integer); defaults to 1.  If \code{NULL}, then a random seed will not be set.
}
  \item{parallel}{
A character string specifying optional parallelization.  Use \code{"seq"} for sequential computation (the default).  Use \code{"multicore"} to perform the k-fold (internal) cross-validation step as well as the learning across subsets in parallel over all available cores.  Or \code{parallel} can be a snow cluster object.  Both parallel options use the built-in functionality of the core "parallel" package. 
}
}

\value{

\item{subfits}{
A list of predictive models, each of which are fit on a subset of the (rows of) data, \code{x}.  For \code{learnControl$multiType="crossprod"}, the length of this list is equal to the number of subsets times the number of learners in the \code{learner} argument.  For \code{learnControl$multiType="divisor"}, the length of this list is equal to the number of subsets.
}
\item{metafit}{
The predictive model which is learned by regressing \code{y} on \code{Z} (see description of \code{Z} below).  The type of model is specified using the \code{metalearner} argument.
}
\item{subpred}{
A data.frame with the predicted values from each sublearner algorithm for the rows in \code{newx}.  If we have L unique learners and there are J subsets of data, then there will be L x J columns when \code{learnControl$multiType=="crossprod"} (default) and J columns when \code{learnControl$multiType=="divisor"}.
}
\item{pred}{
A vector containing the predicted values from the subsemble for the rows in \code{newX}.
}
\item{Z}{
The Z matrix (the cross-validated predicted values for each sublearner).
}
\item{cvRisk}{
A numeric vector with the k-fold cross-validated risk estimate for each algorithm in learning library. Note that this does not contain the CV risk estimate for the Subsemble, only the individual models in the library.  (Not enabled yet, set to \code{NULL}.)
}
\item{family}{
Returns the \code{family} argument from above.  
}
\item{subControl}{
Returns the \code{subControl} argument from above.
}
\item{cvControl}{
Returns the \code{cvControl} argument from above.
}
\item{learnControl}{
Returns the \code{learnControl} argument from above.
}
\item{subsets}{
The list of subsets, which is a list of vectors of row indicies.  The length of this list equals the number of subsets.
}
\item{subCVsets}{
The list of subsets, further broken down into the cross-validation folds that were used.  Each subset (top level list element) is partitioned into V cross-validation folds.
}
\item{ylim}{
Returns range of \code{y}.
}
\item{seed}{
An integer. Returns \code{seed} argument from above.
}
\item{runtime}{
An list of runtimes for various steps of the algorithm.  The list contains \code{cv}, \code{metalearning}, \code{sublearning} and \code{total} elements.  The \code{cv} element is the time it takes to create the \code{Z} matrix (see above).  The \code{metalearning} element is the training time for the metalearning step.  The \code{sublearning} element is a list of training times for each of the models in the ensemble.  The time to run the entire \code{subsemble} function is given in \code{total}.
}

}

\references{
LeDell, E. (2015) Scalable Ensemble Learning and Computationally Efficient Variance Estimation (Doctoral Dissertation).  University of California, Berkeley, USA.\cr
\url{http://www.stat.berkeley.edu/~ledell/papers/ledell-phd-thesis.pdf}\cr
\cr
Stephanie Sapp, Mark J. van der Laan & John Canny. Subsemble: An ensemble method for combining subset-specific algorithm fits.  Journal of Applied Statistics, 41(6):1247-1259, 2014\cr
\url{http://www.tandfonline.com/doi/abs/10.1080/02664763.2013.864263}\cr
\url{https://biostats.bepress.com/ucbbiostat/paper313}
}
\author{
Erin LeDell  \email{ledell@berkeley.edu} 
}
\note{
Notice to users: The interface (function arguments/values) may be subject to change prior to version 1.0.0.
}


\seealso{
\code{\link[SuperLearner:listWrappers]{listWrappers}}, \code{\link[SuperLearner:SuperLearner]{SuperLearner}}
}
\examples{
\dontrun{
# Load some example data.

library(subsemble)
library(cvAUC)  # >= version 1.0.1
data(admissions)

# Training data.
x <- subset(admissions, select = -c(Y))[1:400,]
y <- admissions$Y[1:400]

# Test data.
newx <- subset(admissions, select = -c(Y))[401:500,]
newy <- admissions$Y[401:500]


# Set up the Subsemble.

learner <- c("SL.randomForest", "SL.glm")
metalearner <- "SL.glm"
subsets <- 2


# Train and test the model.
# With learnControl$multiType="crossprod" (the default), 
# we ensemble 4 models (2 subsets x 2 learners).

fit <- subsemble(x = x, y = y, newx = newx, family = binomial(), 
                 learner = learner, metalearner = metalearner,
                 subsets = subsets)


# Evaulate the model by calculating AUC on the test set.

auc <- AUC(predictions = fit$pred, labels = newy)
print(auc)  # Test set AUC is: 0.937


# We can also use the predict method to generate predictions on new data afterwards.

pred <- predict(fit, newx)
auc <- AUC(predictions = pred$pred, labels = newy)
print(auc)  # Test set AUC is: 0.937


# Modify the learnControl argument and train/eval a new Subsemble.
# With learnControl$multiType="divisor", 
# we ensemble only 2 models (one for each subset).

fit <- subsemble(x = x, y = y, newx = newx, family = binomial(), 
                 learner = learner, metalearner = metalearner,
                 subsets = subsets,
                 learnControl = list(multiType="divisor"))

auc <- AUC(predictions = fit$pred, labels = newy)
print(auc)  # Test set AUC is: 0.922


# An example using a single learner.
# In this case there are 3 subsets and 1 learner,
# for a total of 3 models in the ensemble.

learner <- c("SL.randomForest")
metalearner <- "SL.glmnet"
subsets <- 3

fit <- subsemble(x = x, y = y, newx = newx, family = binomial(),
                 learner = learner, metalearner = metalearner,
                 subsets = subsets)
                 
auc <- AUC(predictions = fit$pred, labels = newy)
print(auc)  # Test set AUC is: 0.925


# An example using the full data (i.e. subsets = 1).  
# Here, we have an ensemble of 2 models (one for each of the 2 learners).
# This is equivalent to the Super Learner algorithm.

learner <- c("SL.randomForest", "SL.glm")
metalearner <- "SL.glm"
subsets <- 1

fit <- subsemble(x = x, y = y, newx = newx, family = binomial(), 
                 learner = learner, metalearner = metalearner,
                 subsets = subsets)
                 
auc <- AUC(predictions = fit$pred, labels = newy)
print(auc)  # Test set AUC is: 0.935


# Multicore subsemble via the "parallel" package.
# To perform the cross-validation and training steps using all available cores, 
# use the parallel = "multicore" option.

learner <- c("SL.randomForest", "SL.glm")
metalearner <- "SL.glm"
subsets <- 2

fit <- subsemble(x = x, y = y, newx = newx, family = binomial(),
                 learner = learner, metalearner = metalearner,
                 subsets = subsets, parallel = "multicore")

auc <- AUC(predictions = fit$pred, labels = newy)
print(auc)  # Test set AUC is: 0.937


# Parallel subsemble via the "doSNOW" package.
# To perform the cross-validation and training sets using a SNOW cluster,
# set up a SNOW cluster and pass the cluster object to subsemble.
# If using the "SOCK" cluster type, make sure to export any functions
# that are needed on the cluster nodes, like the algorithm wrapper functions.

library(doSNOW)

nodes <- 4
cl <- makeCluster(nodes, type = "SOCK")
registerDoSNOW(cl)
clusterExport(cl, c(learner, metalearner)) 

fit <- subsemble(x = x, y = y, newx = newx, family = binomial(),
                 learner = learner, metalearner = metalearner,
                 subsets = subsets, parallel = cl)
stopCluster(cl)                 
                 
auc <- AUC(predictions = fit$pred, labels = newy)
print(auc)  # Test set AUC is: 0.937


# More information at: https://github.com/ledell/subsemble
}
}
\keyword{models}
