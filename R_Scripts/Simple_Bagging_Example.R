#Simple Bagging Example
#load up the library 
library(ipred)
library(caret)
set.seed(300)

#create ensemble using decision tree with 25 trees
mybag <- bagging(default ~. , data = credit, nbagg = 25)

credit_pred <- predict(mybag, credit)

table(credit_pred, credit$default, dnn = c("predicted", "actual"))

#Bagging with 10 fold CV
set.seed(300)

#control object
ctrl <- trainControl(method = "cv", number = 10)

#train the model
credit_train <- train(default ~. , data = credit, method = "treebag", trControl = ctrl)

#building the predictor
credit_pred <- predict(credit_train, credit)

#crosstable
table(credit_pred, credit$default)