#L1 penalty in action
# performing LASSO and comparing it to OLS

#Libraries used
library(glmnet)
library(MASS)

set.seed(1234)
X <- mvrnorm(n = 200, mu = c(0, 0, 0, 0, 0),
             Sigma = matrix(c(
               1, .9999, .99, .99, .10,
               .9999, 1, .99, .99, .10,
               .99, .99, 1, .99, .10,
               .99, .99, .99, 1, .10,
               .10, .10, .10, .10, 1
             ), ncol = 5))

y <- rnorm(200, 3 + X %*% matrix(c(1, 1, 1, 1, 0)), .5)

# Fit an OLS over the first 100 cases
m.ols <- lm(y[1:100] ~ X[1:100, ])


#using cross validation to get lambda, hypertune from there
#alpha = 1 is LASSO, alpha = 0 is ridge
m.lasso.cv <- cv.glmnet(X[1:100, ], y[1:100], alpha = 1)

#plot to see the MSE for a given lambda, numbers above plot are number of variables not zero
plot(m.lasso.cv)

#always plot and observe
#when penalty gets too high, cv model error gets high
#lasso seems to do well with LOW LAMBDA values, indicating that it is not that useful
#does not improve generalizability by dropping more variables 
#Hence, does not help improve the out of sample performance

#omparing coefficients of OLS and LASSO

compare <- cbind(OLS = coef(m.ols), LASSO = coef(m.lasso.cv))
compare

#Notice that the OLS coefficients are noisier and LASSO predictor 5 coef is penalized to zero
#Bear in mind that the true intercept and coefficients are 3,1,1,1,0
#OLS has too low values for predictors and LASSO is more accurate for each


#L2 Regularization, Ridge
# use cv.glmnet() and alpha = 0
m.ridge.cv <- cv.glmnet(X[1:100, ], y[1:100], alpha = 0)

plot(m.ridge.cv)

#again, clear that when penalty gets too high, the MSE of CV increases
#seems to do well with lower lambda values, indicating that L2 might not be useful
# to improve generalizability or improve out of sample performance

#combining OLS, L1, L2

cbind(OLS = coef(m.ols), LASSO = coef(m.lasso.cv)[ ,1], RIDGE = coef(m.ridge.cv)[, 1], TRUE_VALUES = c(3,1,1,1,1,0))

#all ridge values are slightly shrunken but closest to true values among all
# despite not shrinking the last term to 0 