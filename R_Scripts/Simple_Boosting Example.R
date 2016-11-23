library(adabag) #for boosting
set.seed(300)

#building the classifier 
m_adaboost <- boosting(default ~., data = credit)

#building the predictor
p_adaboost <- predict(m_adaboost, credit)

#ConfusionMatrix
p_adaboost$confusion

#boosting with Cross Validation (CV)
adaboost_cv <- boosting.cv(default ~., data = credit)

#Confusion Matrix
adaboost_cv$confusion