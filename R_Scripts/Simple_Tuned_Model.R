#Creating a simple tuned model
library(caret)
library(C50)
set.seed(300)

#load the csv file
credit <- read.csv(file = "Machine-Learning-with-R-datasets-master/credit.csv", stringsAsFactors = TRUE)

#replace numbers with strings as class levels
credit$default <- ifelse(credit$default == 1, "yes", "no")

#change strings to factors, better for classifiers
credit$default <- as.factor(credit$default)

#training the model using train() from caret
m <- train(default ~., data = credit, method = "C5.0")

#view which parameters give the best candidate model
m

#building the predictor
p <- predict(m, credit)

#confusion matrix
table(p, credit$default)
