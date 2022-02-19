Ridge Regression and Lasso
================
Lin Yang

``` r
library(ISLR) #only for data
library(glmnet) #for regularized regression, can do all three models
library(caret)
library(corrplot)
library(plotmo)#plotmodel, generate trace plot
```

Predict a baseball player’s salary on the basis of various statistics
associated with performance in the previous year. Use `?Hitters` for
more details.
