#Random Forest vs C5.0 Example
library(randomForest)
library(caret)
set.seed(300)

#test rf
rf <- randomForest(default ~., data = credit)

#want to train a Random Forest with 10-Fold CV repeated 10 times with .mtry values of 2,4,8,16
#create a control object to guide train()
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

#create a grid that will be tuned with .mtry parameters, number of randomly selected feature split
#for each tree
grid_rf <- expand.grid(.mtry = c(2, 4, 8, 16))

#building the classifier for rf
m_rf <- train(default ~., data = credit, method = "rf", metric = "Kappa", trControl = ctrl, tuneGrid = grid_rf)



#C5.0 comparison using same control with trials of 10,20,30, and 40
#create grid for tuning
grid_c50 <- expand.grid(.model = "tree", .trials = c(10, 20, 30, 40), .winnow = "FALSE")
set.seed(300)
m_c50 <- train(default ~ ., data = credit, method = "C5.0", metric = "Kappa", trControl = ctrl, tuneGrid = grid_c50)

#compare both

m_rf

m_c50
