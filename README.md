# subsemble

The `subsemble` package is an R implementation of the Subsemble algorithm. Subsemble is a general subset ensemble prediction method, which can be used for small, moderate, or large datasets. Subsemble partitions the full dataset into subsets of observations, fits a specified underlying algorithm on each subset, and uses a unique form of k-fold cross-validation to output a prediction function that combines the subset-specific fits. An oracle result provides a theoretical performance guarantee for Subsemble.

[Stephanie Sapp](https://www.linkedin.com/in/sappstephanie), [Mark J. van der Laan](http://www.stat.berkeley.edu/~laan/index.html) & [John Canny](https://en.wikipedia.org/wiki/John_Canny). Subsemble: An ensemble method for combining subset-specific algorithm fits. *Journal of Applied Statistics*, 41(6):1247-1259, 2014.

- Article: [http://www.tandfonline.com/doi/abs/10.1080/02664763.2013.864263](http://www.tandfonline.com/doi/abs/10.1080/02664763.2013.864263)
- Preprint: [https://biostats.bepress.com/ucbbiostat/paper313](https://biostats.bepress.com/ucbbiostat/paper313)


## Install subsemble

You can install:

-   the latest released version from CRAN with

    ```r
    install.packages("subsemble")
    ```

-   the latest development version from GitHub with

    ```r
    if (packageVersion("devtools") < 1.6) {
      install.packages("devtools")
    }
    devtools::install_github("ledell/subsemble")
    ```

## Using subsemble

Here are some examples of how to use the `subsemble` package to do various types of learning tasks.  These examples are also part of the `subsemble` function documentation in the R package.

Load some example binary outcome data to use in all the examples below.

```r
library(subsemble)
library(cvAUC)  # >= version 1.0.1
data(admissions)  # From cvAUC package

# Training data.
x <- subset(admissions, select = -c(Y))[1:400,]
y <- admissions$Y[1:400]

# Test data.
newx <- subset(admissions, select = -c(Y))[401:500,]
newy <- admissions$Y[401:500]
```

### Set up the Subsemble

To set up a Subsemble, the user must decide on a base learner library (specified in the `learner` argument) and a metalearning algorithm (specified in the `metalearner` argument).  

The Subsemble below uses only two base learners -- a Random Forest and a GLM.  Both `"SL.randomForest"` and `"SL.glm"` are algorithm wrapper functions from the [SuperLearner](https://github.com/ecpolley/SuperLearner) R package.  

```r
learner <- c("SL.randomForest", "SL.glm")
metalearner <- "SL.glm"
subsets <- 2
```

To customize the base learners, you can wrap these functions with another function, passing in any non-default arguments.  For example, we can create a Random Forest with `ntree = 1500` and `mtry = 2` as follows:

```r
SL.randomForest.1 <- function(..., ntree = 1500, mtry = 2) {
  SL.randomForest(..., ntree = ntree, mtry = mtry)	
}
```
This custom base learner can be included in the learner library as follows:
```r
learner <- c("SL.randomForest.1", "SL.glm")
```

In the `subsets` argument, the user can specify the number of random partitions of the training data.  For instance, if `subsets = 3`, then the training observations will be split into three roughly equal-sized training subsets.  Alternatively, the subsets can be explicitly defined by specifying `subsets` as a list of length `J` (for J subsets).  This will be a list of vector of of row indices.


### Train and test the Subsemble

The `subsemble` function has one argument that is specific to the learning of the ensemble, and that is a list called `learnControl`.  Currently, there is just one element in this list, `multiType`, which defines the "type" of subsemble multi-algorithm learning.  The default, `learnControl[["multiType"]] = "crossprod"`, stands for the "cross-product" type.  This means that every subset is trained with every learner in the base library.  In this example, the Subsemble will be a combination of 4 models (2 subsets x 2 learners).

```r
learner <- c("SL.randomForest", "SL.glm")
metalearner <- "SL.glm"
subsets <- 2

fit <- subsemble(x = x, y = y, newx = newx, family = binomial(), 
                 learner = learner, metalearner = metalearner,
                 subsets = subsets)
```

Since the `newx` argument was not `NULL` in the command above, the `subsemble` function will automatically generate the predicted values for the test set.  The ensemble predicted values for the test set can be found in the `fit$pred` object.  We can estimate model performance by calculating AUC on the test set.

```r
auc <- AUC(predictions = fit$pred, labels = newy)
print(auc)  # Test set AUC is: 0.937
```

By default, `newx` will be `NULL` and will return the predicted values for the training data.  If `newx = NULL` in the statement above, we could use the `predict` method to generate predictions on new data after training the subsemble fit.

```r
pred <- predict(fit, newx)

auc <- AUC(predictions = pred$pred, labels = newy)
print(auc)  # Test set AUC is: 0.937
```

#### Non-default variations

Next, we can modify the `learnControl` argument and then train and test a new Subsemble and compare the results to the default model settings.  With `learnControl[["multiType"] = "divisor"`, we ensemble only 2 models (one for each subset).  In the "divisor" type, the number of models that make up the subsemble is always the same as the number of subsets.  If the number of unique learners is a divisor of the number of subsets (e.g. `length(learner) == 2` and `subsets = 8`), then the base learning algorithms will be recycled as neccessary to get a total of 8 models in the Subsemble.

The "divisor" type of multi-algorithm subsemble will train faster (since there are fewer constituent models), but will likely have worse model performance than the default "cross-product" method.  Therefore, you may choose to use the "divisor" method for rapid testing / exploration, but not neccessarily for your final ensemble.  

```r
fit <- subsemble(x = x, y = y, newx = newx, family = binomial(), 
                 learner = learner, metalearner = metalearner,
                 subsets = subsets,
                 learnControl = list(multiType = "divisor"))

auc <- AUC(predictions = fit$pred, labels = newy)
print(auc)  # Test set AUC is: 0.922
```

The `subsemble` function can also be used with a single base learner.  In this example, there are 3 subsets and 1 learner, for a total of 3 models in the ensemble.

```r
learner <- c("SL.randomForest")
metalearner <- "SL.glmnet"
subsets <- 3

fit <- subsemble(x = x, y = y, newx = newx, family = binomial(),
                 learner = learner, metalearner = metalearner,
                 subsets = subsets)
                 
auc <- AUC(predictions = fit$pred, labels = newy)
print(auc)  # Test set AUC is: 0.925
```

#### Super Learner algorithm

An ensemble can also be created using the full training data by setting `subsets = 1`.  This is equivalent to the [Super Learner](http://biostats.bepress.com/ucbbiostat/paper266/) algorithm.  In the example below, we have an ensemble of 2 models (one for each of the 2 learners).

```r
learner <- c("SL.randomForest", "SL.glm")
metalearner <- "SL.glm"
subsets <- 1

fit <- subsemble(x = x, y = y, newx = newx, family = binomial(), 
                 learner = learner, metalearner = metalearner,
                 subsets = subsets)
                 
auc <- AUC(predictions = fit$pred, labels = newy)
print(auc)  # Test set AUC is: 0.935
```

### Parallel training
One of the benefits to using Subsemble (as opposed to other ensemble algorithms) is that the learning process can be broken up into smaller learning problems that can be easily parallelized across multiple cores or nodes.  This is especially useful in cases where there is not enough memory on your machine(s) to train an ensemble on the full training set.  

#### Multicore Subsemble
To perform the cross-validation and training steps in parallel using all available cores, use the `parallel = "multicore"` option.

```r
learner <- c("SL.randomForest", "SL.glm")
metalearner <- "SL.glm"
subsets <- 2

fit <- subsemble(x = x, y = y, newx = newx, family = binomial(),
                 learner = learner, metalearner = metalearner,
                 subsets = subsets, parallel = "multicore")

auc <- AUC(predictions = fit$pred, labels = newy)
print(auc)  # Test set AUC is: 0.937
```

#### SNOW Subsemble 
To perform the cross-validation and training sets using a [SNOW](http://cran.r-project.org/web/packages/snow/index.html) cluster, the user will need to set up a SNOW cluster and pass the cluster object to the `subsemble` function via the `parallel` argument.  If using the `"SOCK"` cluster type, make sure to export any functions that are needed on the cluster nodes, like the algorithm wrapper functions.

This example uses the [doSNOW](http://cran.r-project.org/web/packages/doSNOW/index.html) library to create a 4-node cluster.

```r
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
```

