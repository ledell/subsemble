# control functions for subsemble
# These are modified versions of the control.R functions from SuperLearner
# Original version of function created by Eric Polley on 2011-01-03.

.cv_control <- function(V = 10L, stratifyCV = TRUE, shuffle = TRUE){
  # Parameters that control the CV process
  # Output used in SuperLearner::CVFolds
  
  # Make sure V is an integer
  V <- as.integer(V)
  if(!is.logical(stratifyCV)) {
    stop("'stratifyCV' must be logical")
  }
  if(!is.logical(shuffle)) {
    stop("'shuffle' must be logical")
  }  
  return(list(V = V, stratifyCV = stratifyCV, shuffle = shuffle))
}

.sub_control <- function(J = 3L, stratifyCV = TRUE, shuffle = TRUE, supervision = NULL){
  # Parameters that control the data partitioning process
  # Output used in SuperLearner::CVFolds  
  
  # J is the number of unique data partitions/subsets
  ctrl <- .cv_control(V=J, stratifyCV=stratifyCV, shuffle=shuffle)
  if (!is.null(supervision)){
    stop("Supervised Subsemble is not yet implemented.  Check back in a future release.")
  }
  ctrl[["supervision"]] <- supervision
  return(ctrl)
}

.learn_control <- function(multiType = "crossprod"){
  # Parameters that control the learning process
  
  # If there are multiple learners, "crossprod" will create
  # an ensemble of K models, where K = J x length(learner)
  if (!(multiType %in% c("crossprod","divisor"))){
    stop("'multiType' must be equal to 'crossprod' or 'divisor'") 
  }
  return(list(multiType = multiType))
}

.gen_control <- function(saveFits = TRUE){
  # General control parameters
  return(list(saveFits = saveFits))
}

