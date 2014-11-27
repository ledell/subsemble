subsemble <- function(x, y, newx = NULL, family = gaussian(), 
                      learner, metalearner = "SL.glm", subsets = 3, subControl = list(), 
                      cvControl = list(), learnControl = list(), genControl = list(),
                      id = NULL, obsWeights = NULL, seed = 1, parallel = "seq")
{
    
    starttime <- Sys.time()
    runtime <- list()
    
    N <- dim(x)[1L]      #Number of observations in training set
    ylim <- range(y)     #Used to enforce bounds
    row.names(x) <- 1:N  #.subFit function requires row names = 1:N
    if (is.null(newx)) {
        newx <- x
    }

    # Update control args by filling in missing list elements
    subControl <- do.call(".sub_control", subControl)
    cvControl <- do.call(".cv_control", cvControl)
    learnControl <- do.call(".learn_control", learnControl)
    genControl <- do.call(".gen_control", genControl)
    # If not a binary outcome, force stratifyCV = FALSE
    if (length(unique(y)) != 2) {
      subControl$stratifyCV = FALSE
      cvControl$stratifyCV = FALSE
    }
    
    # Parse 'subsets' argument
    if (is.numeric(subsets) && length(subsets)==1 && subsets%%1==0) {
        #If subsets is an integer indicating the # of subsets,
        #partition the indices according to subControl
        subControl$V <- subsets
        if (is.numeric(seed)) set.seed(seed)  #If seed given, set seed prior to next step
        subsets <- CVFolds(N=N, id=NULL, Y=y, cvControl=subControl)
    } else if (length(subsets)==N) {
        #If subsets is a vector of subset labels, create list of index vectors
        subsets <- lapply(1:length(unique(subsets)), function(j) which(subsets==j))
        names(subsets) <- as.character(seq(length(subsets)))
        subControl$V <- length(subsets)
    } else if (is.list(subsets) && identical(seq(N), as.integer(sort(unlist(subsets))))) {
        #If subsets is a list of index vectors (ie. user specified partition of the rows of x),
        #force indices to integer type
        subsets <- lapply(subsets, function(ll) as.integer(ll))
        names(subsets) <- as.character(seq(length(subsets)))
        subControl$V <- length(subsets)
    } else {
        stop("'subsets' must be either an integer, a vector of subset labels, or a list of index vectors") 
    }
    
    J <- subControl$V     #Number of subsets (partitions of x)
    V <- cvControl$V      #Number of CV folds
    L <- length(learner)  #Number of distinct learners
    if (J==1) {
      message("Setting J=1 will fit an ensemble on the full data; this is equivalent to the Super Learner algorithm.")
    }

    # Validate learner and metalearner arguments
    multilearning <- ifelse(L>1, TRUE, FALSE)
    if (learnControl$multiType == "divisor" | L==1) {
      if (J %% L == 0) {
        #Recycle learner list as neccessary to get J total learners
        learner <- rep(learner, J/L)
      } else {
        message("The length of 'learner' must be a divisor of the number of subsets.")
      }      
    }
    if (length(metalearner)>1 | !is.character(metalearner)) {
        stop("The 'metalearner' argument must be a string, specifying the name of a SuperLearner wrapper function.")
    }
    if (sum(!sapply(learner, exists))>0) {
        stop("'learner' function name(s) not found.")
    }
    if (!exists(metalearner)) {
        stop("'metalearner' function name not found.")
    }
    #.check.SL.library(learner)
    #.check.SL.library(metalearner)
    
    # Validate remaining arguments  
    if (is.character(family)) 
        family <- get(family, mode = "function", envir = parent.frame())
    if (is.function(family)) 
        family <- family()
    if (is.null(family$family)) {
        print(family)
        stop("'family' not recognized")
    }    
    if (is.null(id)) {
         id <- seq(N)
    } else if (length(id)!=N) {
        stop("If specified, the 'id' argument must be a vector of length nrow(x)")
    }
    if (is.null(obsWeights)) {
        obsWeights <- rep(1, N)
    }
    if (inherits(parallel, "character")) {
      if (!(parallel %in% c("seq","multicore"))) {
        stop("'parallel' must be either 'seq' or 'multicore' or a snow cluster object")
      }
    } else if (!inherits(parallel, "cluster")) {
        stop("'parallel' must be either 'seq' or 'multicore' or a snow cluster object")
    }
    if (parallel!="seq"){
        require(parallel)
        ncores <- detectCores()
    }
    
            
    # For each J subset, assign indices to CV folds, such that each fold contains ~N/J points
    .subCVsplit <- function(idxs, y, cvControl) {
        #Splits a list of idxs into V folds, as directed by cvControl 
        folds <- CVFolds(N=length(idxs), id=NULL, Y=y[idxs], cvControl=cvControl)
        subfolds <- lapply(X=folds, FUN=function(ll) idxs[ll]) 
        return(subfolds)
    }
    if (is.numeric(seed)) set.seed(seed)  #If seed given, set seed prior to next step
    subCVsets <- lapply(X=subsets, FUN=.subCVsplit, y=y, cvControl=cvControl)

    # Helper functions for .make_Z_l:
    # Identify the (j,v)th training indices,
    # then train a model and generate predictions on the test set
    .subFun <- function(j, v, subCVsets, y, x, family, learner, obsWeights, seed) {
        idx.train <- setdiff(unlist(subCVsets[[j]]), subCVsets[[j]][[v]])
        idx.test <- unlist(lapply(subCVsets, function(ll) ll[[v]]), use.names=FALSE)  #CV test idxs
        if (is.numeric(seed)) set.seed(seed)  #If seed is specified, set seed prior to next step
        fit <- match.fun(learner[j])(Y=y[idx.train], X=x[idx.train,], newX=x[idx.test,],
                                  family=family, obsWeights=obsWeights[idx.train])
        if (!is.null(names(fit$pred)) && is.vector(fit$pred)) {
            if (!identical(as.integer(names(fit$pred)), unlist(sapply(1:J, function(ll) subCVsets[[ll]][[v]], simplify=FALSE)))) {
                stop("names not identical")
            }
        }
        preds <- as.vector(fit$pred)  #Force a vector; some wrappers return a 1-col df (ie. SL.glmnet)
        #Make sure preds are identified by the original x row index
        names(preds) <- as.vector(unlist(sapply(subCVsets, function(ll) ll[[v]], simplify=FALSE)))
        return(preds)
    }            
    # Return a list of length J containing the j^th predictions for fold v 
    # Operates on a single learner
    .cvFun <- function(v, subCVsets, y, xmat, family, learner, obsWeights, seed) {
        preds <- lapply(X=1:J, FUN=.subFun, v=v, subCVsets=subCVsets, y=y, x=xmat, family=family,
                      learner=learner, obsWeights=obsWeights, seed=seed)
        return(preds)
    }
    # Create a N x J matrix of cross-validated predicted values
    .subPreds <- function(j, cvRes) {
      subRes <- unlist(cvRes[j,])  #Get all N predictions for col j of Z_l
      return(subRes[as.character(seq(length(subRes)))])  #Reorder by observation index i
    }
    
    # Generate the CV predicted values for a single learner across all subsets
    .make_Z_l <- function(V, subCVsets, y, x, family, learner, obsWeights, parallel, seed) {
      if (inherits(parallel, "cluster")) {
        #If the parallel object is a snow cluster
        cvRes <- parSapply(cl=parallel, X=1:V, FUN=.cvFun, subCVsets=subCVsets, y=y, xmat=x, family=family,
                           learner=learner, obsWeights=obsWeights, seed=seed)  #cvRes is a J x V matrix         
      } else if (parallel=="multicore") {
        cl <- makeCluster(min(ncores,V), type="FORK")  #May update in future to avoid copying all objects in memory
        cvRes <- parSapply(cl=cl, X=1:V, FUN=.cvFun, subCVsets=subCVsets, y=y, xmat=x, family=family,
                           learner=learner, obsWeights=obsWeights, seed=seed)  #cvRes is a J x V matrix
        stopCluster(cl)
      } else {
        cvRes <- sapply(X=1:V, FUN=.cvFun, subCVsets=subCVsets, y=y, xmat=x, family=family,
                        learner=learner, obsWeights=obsWeights, seed=seed)  #cvRes is a J x V matrix
      }
      if (class(cvRes)!="matrix") {
        cvRes <- t(as.matrix(cvRes))
      } 
      Z <- as.data.frame(sapply(1:J, .subPreds, cvRes=cvRes))
      return(Z)
    }
  
    # If multilearning, assign unique names to distinct learners
    # For example, if learner = c("SL.randomForest","SL.randomForest")
    if (multilearning && length(learner)!=length(unique(learner))) {
      if (!is.null(seed)) warning("Repeating identical learner wrappers in the learning library
        when the seed is not NULL will cause these sublearners
        to produce identical submodels (not recommended).")
      lnames <- learner
      for (x in learner) {
        if (sum(learner==x)>1) {
          for (i in seq(sum(learner==x))) {
            idxs <- which(learner==x)
            lnames[idxs[i]] <- paste(learner[idxs[i]], i, sep=".")
          }
        }
      }
    } else if (multilearning){
      lnames <- learner
    } else {
      lnames <- unique(learner)
    }
    
    # Create the Z matrix of cross-validated predictions using preds from each sub-model
    if (multilearning && learnControl$multiType=="crossprod") {      
      # Case of multiple learners (and "crossprod" multiType)
      # Each learner creates an N x J matrix Z_l
      # The final Z matrix of dimension N x (J x L) is the column-wise concatenation of the Z_l matrices
      multilearner <- sapply(learner, function(ll) list(rep(ll,J)))
      
      runtime$cv <- system.time(Z <- do.call("cbind", sapply(X=multilearner, FUN=.make_Z_l, V=V, subCVsets=subCVsets, 
                                   y=y, x=x, family=family, obsWeights=obsWeights, 
                                   parallel=parallel, seed=seed, simplify=FALSE)), gcFirst=FALSE)
      learner <- unlist(multilearner)  #expand learner object to size L x J = # of submodels
    } else {
      # Case of single learner or "divisor" multiType 
      runtime$cv <- system.time(Z <- .make_Z_l(V=V, subCVsets=subCVsets, y=y, x=x, 
                  family=family, learner=learner,
                  obsWeights=obsWeights, seed=seed, parallel=parallel), gcFirst=FALSE)
    }
    if (learnControl$multiType=="crossprod") {
      modnames <- as.vector(sapply(lnames, function(ll) paste(ll, paste("J", seq(J), sep=""), sep="_")))
    } else { #learnControL$multiType == "divisor"
      modnames <- sapply(seq(J), function(i) sprintf("%s_J%s", rep(lnames, J/length(lnames))[i], i))
    }
    names(Z) <- modnames
    row.names(Z) <- row.names(x)  #(Might want to use original x row names instead of 1:N)
    Z[Z < ylim[1]] <- ylim[1]  #Enforce bounds
    Z[Z > ylim[2]] <- ylim[2]


    # Metalearning: Regress y onto Z to learn optimal combination of submodels
    if (is.numeric(seed)) set.seed(seed)  #If seed given, set seed prior to next step
    runtime$metalearning <- system.time(metafit <- match.fun(metalearner)(Y=y, X=Z, newX=Z, 
                                                    family=family, id=id, obsWeights=obsWeights), gcFirst=FALSE) 

    # Train a model on the entire j^th subset of x and generate preds for newx
    .fitFun <- function(m, subCVsets, y, x, newx, family, learner, obsWeights, seed) {
        J <- length(subCVsets)  #Same J as elsewhere
        j <- ifelse((m %% J)==0, J, (m %% J))  #The j subset that m is associated with
        idx.train <- unlist(subCVsets[[j]])
        if (is.numeric(seed)) set.seed(seed)  #If seed given, set seed prior to next step
        fit <- match.fun(learner[m])(Y=y[idx.train], X=x[idx.train,], newX=newx,
                                  family=family, obsWeights=obsWeights[idx.train])
        return(fit)
    }
    # Wrapper function for .fitFun to record system.time
    .fitWrapper <- function(m, subCVsets, y, xmat, newx, family, learner, obsWeights, seed) {
      fittime <- system.time(fit <- .fitFun(m, subCVsets, y, xmat, newx, family, 
                                            learner, obsWeights, seed), gcFirst=FALSE)
      return(list(fit=fit, fittime=fittime))
    }
    
    # Fit a final model (composed of M = J x L models) to be saved
    M <- ncol(Z)
    if (inherits(parallel, "cluster")) {
      #If the parallel object is a snow cluster
      sublearners <- parSapply(cl=parallel, X=1:M, FUN=.fitWrapper, subCVsets=subCVsets, y=y, xmat=x, newx=newx,
                               family=family, learner=learner, obsWeights=obsWeights, seed=seed, simplify=FALSE)    
    } else if (parallel=="multicore") {
        cl <- makeCluster(min(ncores,M), type="FORK") 
        sublearners <- parSapply(cl=cl, X=1:M, FUN=.fitWrapper, subCVsets=subCVsets, y=y, xmat=x, newx=newx,
                             family=family, learner=learner, obsWeights=obsWeights, seed=seed, simplify=FALSE)
        stopCluster(cl)
    } else {
        sublearners <- sapply(X=1:M, FUN=.fitWrapper, subCVsets=subCVsets, y=y, xmat=x, newx=newx,
                          family=family, learner=learner, obsWeights=obsWeights, seed=seed, simplify=FALSE)
    } 
    runtime$sublearning <- lapply(sublearners, function(ll) ll$fittime)
    names(runtime$sublearning) <- modnames
    
    # subpred: Matrix of predictions for newx; one pred column for each m submodels
    subpred <- do.call("cbind", lapply(sublearners, function(ll) ll$fit$pred))
    subpred <- as.data.frame(subpred)
    names(subpred) <- names(Z)
    subpred[subpred < ylim[1]] <- ylim[1]  #Enforce bounds
    subpred[subpred > ylim[2]] <- ylim[2]    

    # Create subsemble predictions as an ensemble of the J x L sublearners
    pred <- predict(metafit$fit, newdata=subpred, family=family)
    pred[pred < ylim[1]] <- ylim[1]  #Enforce bounds
    pred[pred > ylim[2]] <- ylim[2]      

    # List of submodels fits to be saved
    if (genControl$saveFits) {
      subfits <- lapply(sublearners, function(ll) ll$fit$fit)
      names(subfits) <- names(Z)      
    } else {
      subfits = NULL
      metafit = NULL
    }
    runtime$total <- Sys.time() - starttime
    
    # Results
    out <- list(subfits=subfits, 
                metafit=metafit, 
                subpred=subpred,
                pred=pred,
                Z=Z, 
                cvRisk=NULL,  #To do: Calculate cvRisk given some loss function
                family=family, 
                subControl=subControl,
                cvControl=cvControl, 
                learnControl=learnControl, 
                subsets=subsets,
                subCVsets=subCVsets,
                ylim=ylim, 
                seed=seed,
                runtime=runtime)
    class(out) <- "subsemble"
    return(out)
}

