\name{predict.subsemble}
\alias{predict.subsemble}
\title{
Predict method for a 'subsemble' object.
}
\description{
Obtains predictions on a new data set from a \code{\link[subsemble]{subsemble}} fit.  May require the original data, \code{X}, if one of the learner algorithms uses the original data in its predict method.
}
\usage{
\method{predict}{subsemble}(object, newx, x = NULL, y = NULL, ...)
}
\arguments{
  \item{object}{
An object of class 'subsemble', which is returned from the \code{\link[subsemble]{subsemble}} function.
}
  \item{newx}{
The predictor variables for a new (testing) data set. The structure should match \code{x}.
}
  \item{x}{
Original data set used to fit \code{object}.  
}
  \item{y}{
Original outcome used to fit \code{object}. 
}
  \item{\dots}{
Additional arguments passed to the \code{predict.SL.*} functions.
}
}
\details{
If \code{newx} is omitted, the predicted values from \code{object} are returned.  The \code{learner} algorithm needs to have a corresponding prediction function with ``predict'' prefixed onto the algorithm name (e.g. \code{predict.SL.glm} for \code{SL.glm}).  This should be taken care of by the \code{\link[SuperLearner:SuperLearner]{SuperLearner}} package.
}
\value{
\item{pred}{
Predicted values from subsemble fit.
}
\item{subpred}{
A data.frame with the predicted values from each sublearner algorithm for the rows in \code{newx}.  If we have trained M individual models, then there will be M columns.
}
}

\author{
Erin LeDell \email{ledell@berkeley.edu}
}

\seealso{
\code{\link[subsemble:subsemble]{subsemble}}
}
\examples{
# See subsemble documentation for an example.

}
\keyword{models}
