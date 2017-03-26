#Libraries Used
library(h2o)
library(caret) #do not use data.table with h2o due to length confusion
library(reshape2)
#instantiate a h2o cluster
cl <- h2o.init(max_mem_size = "4G", nthreads = 2)

#dataset loading
use.train.x <- read.table(file = "UCI HAR Dataset/train/X_train.txt")
use.test.x <- read.table(file = "UCI HAR Dataset/test/X_test.txt")

#labels for train and test set
use.train.y <- read.table(file = "UCI HAR Dataset/train/y_train.txt")[[1]]
use.test.y <- read.table(file = "UCI HAR Dataset/test/y_test.txt")[[1]]

#ALWAYS remember to convert into factor for a CLASSIFICATION problem, else treated
#as a regression
#combining columns into a whole dataset
use.train <- cbind(use.train.x, Outcome = factor(use.train.y))
use.test <- cbind(use.test.x, Outcome = factor(use.test.y))

#Mapping table of outcome levels
use.labels <- read.table(file = "UCI HAR Dataset/activity_labels.txt")

#create h2o item in cloud 
h2oactivity.train <- as.h2o(use.train, destination_frame = "h2oactivity.train")
h2oactivity.test <- as.h2o(use.test, destination_frame = "h2oactivity.test")

#to manually generate predictions on new data, do not include testing data in
#the h2o.deeplearning() function, there should be no validation_frame parameter
# DNN with one hidden layer with 50 hidden neurons

mt1 <- h2o.deeplearning(x = colnames(use.train.x), 
                        y = "Outcome",
                        training_frame = h2oactivity.train,
                        activation = "RectifierWithDropout",
                        hidden = c(50),
                        epochs = 10,
                        loss = "CrossEntropy",
                        input_dropout_ratio = .2,
                        hidden_dropout_ratios = c(.5),
                        export_weights_and_biases = TRUE
                    
                        )

#view trained model
#the 28,406 weights came from 561 variables * 50 hidden nodes = 28,050 between input
# and hidden and 50*6 = 300 for hidden and outcome and. 50 biases for hidden neuron
# and 6 biases for outcome
mt1

#extracting features of the model using h2o.deepfeatures()
f <- as.data.frame(h2o.deepfeatures(mt1, h2oactivity.train, 1))

#we get 50 variables due to 50 hidden nodes in that single hidden layer
View(f)

#we can extract weights as well
w1 <- as.matrix(h2o.weights(mt1, 1))
tmp <- as.data.frame(t(w1))
tmp$Row <- 1:nrow(tmp)
tmp <- melt(tmp, id.vars = c("Row"))
#output a heatmap
d3heatmap(w1)

#reconstructing feedforward NN
#input data
d <- as.matrix(use.train[ , -562])

##biases for hidden layer 1
b1 <- as.matrix(h2o.biases(mt1, 1))
b12 <- do.call(rbind, rep(list(t(b1)), nrow(d)))

#manually predict
yhat.h2o <- h2o.predict(mt1, newdata = h2oactivity.test)
View(yhat.h2o[ ,1])
