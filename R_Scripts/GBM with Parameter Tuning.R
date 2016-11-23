#Performing GBM on iris dataset using 10 Fold CV repeated 10 times
#load the relevant libraries
library(caret)
library(gbm)
library(datasets)

#create control object to guide train
ctrl <- trainControl(method = "repeatedCV", number = 10, repeats = 10, verboseIter = TRUE)

# parameter tuning to acheive accuracy of 95.7% with kappa of 0.933
iris_grid <- expand.grid(.n.trees = c(50, 100, 200, 300, 400, 500), .shrinkage =c(0.005, 0.1), .n.minobsinnode = 7, .interaction.depth = 1)

# model training
iris_gbm <- train(Species ~. , data = iris, trControl = ctrl, method = "gbm", metric = "Kappa", tuneGrid = iris_grid)

iris_gbm
