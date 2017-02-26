#libraries used
library(DataExplorer)
library(nnet)

# data loading
seeds <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt")
colnames(seeds) <- c("Area", "Perimeter", "Compactness", "Length", "Width", "Coefficient", "Groove", "Wheat")
View(seeds)
str(seeds)

# EDA
GenerateReport(seeds)

#separate to train and test
seedstrain <- sample(1:210, 147)
seedstest <- setdiff(1:210, seedstrain)

#normalize predicted values, basicalled one hot encoding
ideal <- class.ind(seeds$Wheat)

#train the model
seedsANN <- nnet(seeds[seedstrain, -8], ideal[seedstrain, ], size = 10, softmax = TRUE)

#predict
seedspred <- predict(seedsANN, seeds[seedstest, -8], type = "class")

#accuracy
confusionMatrix(seedspred, seeds[seedstest, ]$Wheat)
