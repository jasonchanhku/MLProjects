#Libraries used
library(h2o)
library(data.table)
library(plotly)
#instantiate h2o cluster
cl <- h2o.init(max_mem_size = "4G", nthreads = 2)

#use fread to read data
d <- fread("YearPredictionMSD.txt", sep = ",")

#distribution of years
plot_ly(d, x=~V1, type = "histogram") %>% 
  layout(title = "Histogram of Song Release Year", 
         xaxis = list(title = "Release Year"), yaxis = list(title = "Count")) 
#distribution shows that lower extreme values may affect influence on the model
#we can exclude small amount of extreme cases by excluding top and bottom 0.5%,
#1% of total data we can do this by checking the quantiles
quantile(d$V1, probs = c(0.005, 0.995))
# shows that it contains years of release from 1957 to 2010, so the data will be trimmed
# in both train and test data from 1957 to 2010

#data trimming
d.train <- d[1:463715][V1 >= 1957 & V1 <= 2010]
d.test <- d[463716:515345][V1 >= 1957 & V1 <= 2010]

#convert training and test dataset for h2o
h2omsd.train <- as.h2o(d.train, destination_frame = "h2omsdtrain")
h2omsd.test <- as.h2o(d.test, destination_frame = "h2omsdtest")

#Benchmark to beat (compared to R^2 of linear regression)
m0 <- lm(V1 ~ . , data = d.train)
summary(m0)$r.squared

#for accuracy in regression problems, use cor
cor(d.test$V1, predict(m0, newdata = d.test))^2
#around 23% of variance explained by lm

#shallow NN with single hidden layer with 50 hidden nodes
#score samples both set to 0 for full coverage of dataset
m1 <- h2o.deeplearning(x = colnames(d)[-1], y = "V1",
                       training_frame = h2omsd.train,
                       validation_frame = h2omsd.test,
                       activation = "RectifierWithDropout",
                       hidden = c(50),
                       epochs = 100,
                       input_dropout_ratio = 0,
                       hidden_dropout_ratios = c(0),
                       score_training_samples = 0,
                       score_validation_samples = 0,
                       diagnostics = TRUE,
                       export_weights_and_biases = TRUE,
                       variable_importances = TRUE)

h2o.scoreHistory(m1)
h2o.r2(m1)

#to improve, use a deep feedforward neural network
m2 <- h2o.deeplearning(
  x = colnames(d)[-1],
  y = "V1",
  training_frame= h2omsd.train,
  validation_frame = h2omsd.test,
  activation = "RectifierWithDropout",
  hidden = c(200, 200, 400),
  epochs = 100,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(.2, .2, .2),
  score_training_samples = 0,
  score_validation_samples = 0,
  diagnostics = TRUE,
  export_weights_and_biases = TRUE,
  variable_importances = TRUE
)

h2o.scoreHistory(m2)
h2o.r2(m2)
#improved r2 in DNN from 0.37 to 40.0

#Prediction Labels, using h2o.predict
yhat <- as.data.frame(h2o.predict(m1, h2omsd.train))
yhat <- cbind(as.data.frame(h2omsd.train[["V1"]]),yhat)
yhat$difference <- yhat$predict - yhat$V1

plot_ly(yhat, x = ~factor(V1), y=~difference, type = "box") %>% layout(title = "Difference of Predicted vs True", xaxis = list(title = ""))

#Variable importance
#extract variable importance based on their contribution to the prediction 
#useful to exclude some variables that dont contribute much
#extract top 10 variable by importance

imp <- as.data.frame(h2o.varimp(m2))
h2o.varimp_plot(m2)

plot_ly(imp, x=~factor(variable, imp$variable), y=~relative_importance, type = "scatter", mode = "markers") %>% layout(title = "Variable Importance Plot", xaxis = list(title = ""))

