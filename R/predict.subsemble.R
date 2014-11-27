predict.subsemble <-
function(object, newx, x = NULL, y = NULL, ...)
{

    if (missing(newx)) {
        out <- list(pred = object$pred, subpred = object$subpred)
        return(out)
    }
    
    J <- length(object$subfits)
    subpred <- as.data.frame(matrix(NA, nrow = nrow(newx), ncol = J))
    for(j in seq(J)) {
        subpred[, j] <- as.vector(do.call('predict', list(object = object$subfits[[j]],
                                              newdata = newx,
                                              family = object$family,
                                              X = x,
                                              Y = y, ...)))
    }
    names(subpred) <- names(object$subfits)
    subpred[subpred < object$ylim[1]] <- object$ylim[1]  #Enforce bounds
    subpred[subpred > object$ylim[2]] <- object$ylim[2]
    row.names(subpred) <- row.names(newx)

    pred <- predict(object$metafit$fit, newdata=subpred, family=family)
    pred[pred < object$ylim[1]] <- object$ylim[1]  #Enforce bounds
    pred[pred > object$ylim[2]] <- object$ylim[2]
    names(pred) <- row.names(newx)

    out <- list(pred = pred, subpred = subpred)
    return(out)
}
