#Training Deep Learning Model
library(h2o)
library(caret)

#create h2o instance
cl <- h2o.init(max_mem_size = "4G", nthreads = 2)

#data setup
digits.train <- read.csv("train.csv")
digits.train$label <- as.factor(digits.train$label)
str(digits.train)

#imports R object to h2o cloud and also store it in a variable
h2odigits <- as.h2o(digits.train, destination_frame = "h2odigits")

#split dataset into training set and test set 
i <- 1:32000
h2odigits.train <- h2odigits[i , ]

itest <- 32001:42000
h2odigits.test <- h2odigits[itest, ]

#store colnames except labels
xnames <- colnames(h2odigits.train)[-1]

#training the first model
# using dropout rectifier, 10 epochs, modest learning rate with 20% dropout for hidden
ex1 <- h2o.deeplearning(x = xnames, y = "label", 
                training_frame = h2odigits.train,
                validation_frame = h2odigits.test,
                activation = "RectifierWithDropout", 
                hidden = c(100), epochs = 10, 
                adaptive_rate = FALSE, rate = 0.001, 
                input_dropout_ratio = 0, 
                hidden_dropout_ratios = c(.2) )

#training the second model 
# just with different learning rate, higher to reduce overfitting. Expect lower accuracy
# but quicker training time.

ex2 <- h2o.deeplearning(x = xnames, y = "label", 
                        training_frame = h2odigits.train,
                        validation_frame = h2odigits.test,
                        activation = "RectifierWithDropout", 
                        hidden = c(100), epochs = 10, 
                        adaptive_rate = FALSE, rate = 0.01, 
                        input_dropout_ratio = 0, 
                        hidden_dropout_ratios = c(.2) )