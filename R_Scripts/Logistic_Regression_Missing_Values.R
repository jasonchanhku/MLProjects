#Detecting Missing Values and Handling Missing Values
#Libraries Used
library(Amelia)
library(ROCR)
library(caret)

#Data loading
train <- read.csv(file = "Titanic/train.csv", na.strings = c(""))

#Check number of missing values
sapply(train, function(x)sum(is.na(x)))

#Which entries are NA ?
which(is.na(train$Age))

#Missing values map
missmap(train, main = "Missing Values Map")

#Subset relevant columns
data <- subset(train, select = c(2,3,5,6,7,8,10,12))

#Taking care of Missing Values
#data$Age[is.na(data$Age)] <- mean(data$Age, na.rm = T)
#age_m <- rpart(Age ~. , data = data[!is.na(data$Age), ], method = "anova")
#age_p <- predict(age_m, data[is.na(data$Age), ])
#data$Age[is.na(data$Age)] <- predict(age_m, data[is.na(data$Age), ])


data <- data[!is.na(train$Embarked), ]
rownames(data) <- NULL

#Model Fitting
raw.data <- train
train <- data[1:800 , ]
test <- data[801:889, ]

#check balance of target variable, stratified
prop.table(table(train$Survived))
prop.table(table(test$Survived))

#contrast
contrasts(train$Sex)

#model training
model <- glm(Survived ~., data = train, family = binomial(link = "logit"))

summary(model)

#Deviance Table 
anova(model, test = "Chisq")

#Building the predictor
pred <- predict(model, test)
pred <- predict(model, test, type = "response")
pred <- ifelse(pred > 0.5, 1, 0)

#Evaluation - accuracy 
err <- mean(pred != test$Survived)
accuracy <- 1 - err
accuracy

#Evaluation - ROC curve
pred_obj <- prediction(pred, test$Survived)
perf <- performance(pred_obj, measure = "tpr", x.measure = "fpr")
plot(perf, main = "ROC Curve", col = "red", lwd = 3)
abline(a = 0 , b = 1, lwd = 2, lty = 2)

#Confusion Matrix
confusionMatrix(pred, test$Survived)