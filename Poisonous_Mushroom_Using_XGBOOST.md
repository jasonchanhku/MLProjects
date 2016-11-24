Predicting Poisonous Mushroom with XGBOOST
================
Jason Chan
November 24, 2016

-   Libraries Used
-   Dataset
    -   Structure
    -   Preview
-   Model Training and Prediction
    -   Tree Visualization
    -   Transformation
-   Model Evaluation
    -   Average Error
    -   Confusion Matrix
    -   ROC Curve
    -   Feature Importance
-   Improving the Model
    -   Parameter Tuning

For the fully functional html version, please visit <http://www.rpubs.com/jasonchanhku/poison>

Libraries Used
==============

``` r
library(xgboost)
library(knitr)
library(caret)
library(ROCR)
library(DiagrammeR)
```

<br />

Dataset
=======

The dataset comes in a `list` where each variable is a `list` containing `label` and `data`. As xgboost can only take in **numeric vectors**, the features have all been **one hot encoded**. Bear in mind that for `label`, it containes the class levels where 0 is "edible" and 1 is "poisonous". The data has already been prepared into train and test as below:

``` r
#data loading
data(agaricus.train, package = "xgboost")
data(agaricus.test, package = "xgboost")
train <- agaricus.train
test <- agaricus.test
```

Structure
---------

``` r
str(train)
```

    ## List of 2
    ##  $ data :Formal class 'dgCMatrix' [package "Matrix"] with 6 slots
    ##   .. ..@ i       : int [1:143286] 2 6 8 11 18 20 21 24 28 32 ...
    ##   .. ..@ p       : int [1:127] 0 369 372 3306 5845 6489 6513 8380 8384 10991 ...
    ##   .. ..@ Dim     : int [1:2] 6513 126
    ##   .. ..@ Dimnames:List of 2
    ##   .. .. ..$ : NULL
    ##   .. .. ..$ : chr [1:126] "cap-shape=bell" "cap-shape=conical" "cap-shape=convex" "cap-shape=flat" ...
    ##   .. ..@ x       : num [1:143286] 1 1 1 1 1 1 1 1 1 1 ...
    ##   .. ..@ factors : list()
    ##  $ label: num [1:6513] 1 0 0 1 0 0 0 1 0 0 ...

Preview
-------

A preview of the dataset is possible, by coercing the list into a matrix. The matrix is bound to be large due to the **one hot encoding**.

``` r
#coercing it into a matrix
mat <- as.matrix(train$data)

#preview of the matrix
kable(data.frame(head(mat)))
```

|  cap.shape.bell|  cap.shape.conical|  cap.shape.convex|  cap.shape.flat|  cap.shape.knobbed|  cap.shape.sunken|  cap.surface.fibrous|  cap.surface.grooves|  cap.surface.scaly|  cap.surface.smooth|  cap.color.brown|  cap.color.buff|  cap.color.cinnamon|  cap.color.gray|  cap.color.green|  cap.color.pink|  cap.color.purple|  cap.color.red|  cap.color.white|  cap.color.yellow|  bruises..bruises|  bruises..no|  odor.almond|  odor.anise|  odor.creosote|  odor.fishy|  odor.foul|  odor.musty|  odor.none|  odor.pungent|  odor.spicy|  gill.attachment.attached|  gill.attachment.descending|  gill.attachment.free|  gill.attachment.notched|  gill.spacing.close|  gill.spacing.crowded|  gill.spacing.distant|  gill.size.broad|  gill.size.narrow|  gill.color.black|  gill.color.brown|  gill.color.buff|  gill.color.chocolate|  gill.color.gray|  gill.color.green|  gill.color.orange|  gill.color.pink|  gill.color.purple|  gill.color.red|  gill.color.white|  gill.color.yellow|  stalk.shape.enlarging|  stalk.shape.tapering|  stalk.root.bulbous|  stalk.root.club|  stalk.root.cup|  stalk.root.equal|  stalk.root.rhizomorphs|  stalk.root.rooted|  stalk.root.missing|  stalk.surface.above.ring.fibrous|  stalk.surface.above.ring.scaly|  stalk.surface.above.ring.silky|  stalk.surface.above.ring.smooth|  stalk.surface.below.ring.fibrous|  stalk.surface.below.ring.scaly|  stalk.surface.below.ring.silky|  stalk.surface.below.ring.smooth|  stalk.color.above.ring.brown|  stalk.color.above.ring.buff|  stalk.color.above.ring.cinnamon|  stalk.color.above.ring.gray|  stalk.color.above.ring.orange|  stalk.color.above.ring.pink|  stalk.color.above.ring.red|  stalk.color.above.ring.white|  stalk.color.above.ring.yellow|  stalk.color.below.ring.brown|  stalk.color.below.ring.buff|  stalk.color.below.ring.cinnamon|  stalk.color.below.ring.gray|  stalk.color.below.ring.orange|  stalk.color.below.ring.pink|  stalk.color.below.ring.red|  stalk.color.below.ring.white|  stalk.color.below.ring.yellow|  veil.type.partial|  veil.type.universal|  veil.color.brown|  veil.color.orange|  veil.color.white|  veil.color.yellow|  ring.number.none|  ring.number.one|  ring.number.two|  ring.type.cobwebby|  ring.type.evanescent|  ring.type.flaring|  ring.type.large|  ring.type.none|  ring.type.pendant|  ring.type.sheathing|  ring.type.zone|  spore.print.color.black|  spore.print.color.brown|  spore.print.color.buff|  spore.print.color.chocolate|  spore.print.color.green|  spore.print.color.orange|  spore.print.color.purple|  spore.print.color.white|  spore.print.color.yellow|  population.abundant|  population.clustered|  population.numerous|  population.scattered|  population.several|  population.solitary|  habitat.grasses|  habitat.leaves|  habitat.meadows|  habitat.paths|  habitat.urban|  habitat.waste|  habitat.woods|
|---------------:|------------------:|-----------------:|---------------:|------------------:|-----------------:|--------------------:|--------------------:|------------------:|-------------------:|----------------:|---------------:|-------------------:|---------------:|----------------:|---------------:|-----------------:|--------------:|----------------:|-----------------:|-----------------:|------------:|------------:|-----------:|--------------:|-----------:|----------:|-----------:|----------:|-------------:|-----------:|-------------------------:|---------------------------:|---------------------:|------------------------:|-------------------:|---------------------:|---------------------:|----------------:|-----------------:|-----------------:|-----------------:|----------------:|---------------------:|----------------:|-----------------:|------------------:|----------------:|------------------:|---------------:|-----------------:|------------------:|----------------------:|---------------------:|-------------------:|----------------:|---------------:|-----------------:|-----------------------:|------------------:|-------------------:|---------------------------------:|-------------------------------:|-------------------------------:|--------------------------------:|---------------------------------:|-------------------------------:|-------------------------------:|--------------------------------:|-----------------------------:|----------------------------:|--------------------------------:|----------------------------:|------------------------------:|----------------------------:|---------------------------:|-----------------------------:|------------------------------:|-----------------------------:|----------------------------:|--------------------------------:|----------------------------:|------------------------------:|----------------------------:|---------------------------:|-----------------------------:|------------------------------:|------------------:|--------------------:|-----------------:|------------------:|-----------------:|------------------:|-----------------:|----------------:|----------------:|-------------------:|---------------------:|------------------:|----------------:|---------------:|------------------:|--------------------:|---------------:|------------------------:|------------------------:|-----------------------:|----------------------------:|------------------------:|-------------------------:|-------------------------:|------------------------:|-------------------------:|--------------------:|---------------------:|--------------------:|---------------------:|-------------------:|--------------------:|----------------:|---------------:|----------------:|--------------:|--------------:|--------------:|--------------:|
|               0|                  0|                 1|               0|                  0|                 0|                    0|                    0|                  0|                   1|                1|               0|                   0|               0|                0|               0|                 0|              0|                0|                 0|                 1|            0|            0|           0|              0|           0|          0|           0|          0|             1|           0|                         0|                           0|                     1|                        0|                   1|                     0|                     0|                0|                 1|                 1|                 0|                0|                     0|                0|                 0|                  0|                0|                  0|               0|                 0|                  0|                      1|                     0|                   0|                0|               0|                 1|                       0|                  0|                   0|                                 0|                               0|                               0|                                1|                                 0|                               0|                               0|                                1|                             0|                            0|                                0|                            0|                              0|                            0|                           0|                             1|                              0|                             0|                            0|                                0|                            0|                              0|                            0|                           0|                             1|                              0|                  1|                    0|                 0|                  0|                 1|                  0|                 0|                1|                0|                   0|                     0|                  0|                0|               0|                  1|                    0|               0|                        1|                        0|                       0|                            0|                        0|                         0|                         0|                        0|                         0|                    0|                     0|                    0|                     1|                   0|                    0|                0|               0|                0|              0|              1|              0|              0|
|               0|                  0|                 1|               0|                  0|                 0|                    0|                    0|                  0|                   1|                0|               0|                   0|               0|                0|               0|                 0|              0|                0|                 1|                 1|            0|            1|           0|              0|           0|          0|           0|          0|             0|           0|                         0|                           0|                     1|                        0|                   1|                     0|                     0|                1|                 0|                 1|                 0|                0|                     0|                0|                 0|                  0|                0|                  0|               0|                 0|                  0|                      1|                     0|                   0|                1|               0|                 0|                       0|                  0|                   0|                                 0|                               0|                               0|                                1|                                 0|                               0|                               0|                                1|                             0|                            0|                                0|                            0|                              0|                            0|                           0|                             1|                              0|                             0|                            0|                                0|                            0|                              0|                            0|                           0|                             1|                              0|                  1|                    0|                 0|                  0|                 1|                  0|                 0|                1|                0|                   0|                     0|                  0|                0|               0|                  1|                    0|               0|                        0|                        1|                       0|                            0|                        0|                         0|                         0|                        0|                         0|                    0|                     0|                    1|                     0|                   0|                    0|                1|               0|                0|              0|              0|              0|              0|
|               1|                  0|                 0|               0|                  0|                 0|                    0|                    0|                  0|                   1|                0|               0|                   0|               0|                0|               0|                 0|              0|                1|                 0|                 1|            0|            0|           1|              0|           0|          0|           0|          0|             0|           0|                         0|                           0|                     1|                        0|                   1|                     0|                     0|                1|                 0|                 0|                 1|                0|                     0|                0|                 0|                  0|                0|                  0|               0|                 0|                  0|                      1|                     0|                   0|                1|               0|                 0|                       0|                  0|                   0|                                 0|                               0|                               0|                                1|                                 0|                               0|                               0|                                1|                             0|                            0|                                0|                            0|                              0|                            0|                           0|                             1|                              0|                             0|                            0|                                0|                            0|                              0|                            0|                           0|                             1|                              0|                  1|                    0|                 0|                  0|                 1|                  0|                 0|                1|                0|                   0|                     0|                  0|                0|               0|                  1|                    0|               0|                        0|                        1|                       0|                            0|                        0|                         0|                         0|                        0|                         0|                    0|                     0|                    1|                     0|                   0|                    0|                0|               0|                1|              0|              0|              0|              0|
|               0|                  0|                 1|               0|                  0|                 0|                    0|                    0|                  1|                   0|                0|               0|                   0|               0|                0|               0|                 0|              0|                1|                 0|                 1|            0|            0|           0|              0|           0|          0|           0|          0|             1|           0|                         0|                           0|                     1|                        0|                   1|                     0|                     0|                0|                 1|                 0|                 1|                0|                     0|                0|                 0|                  0|                0|                  0|               0|                 0|                  0|                      1|                     0|                   0|                0|               0|                 1|                       0|                  0|                   0|                                 0|                               0|                               0|                                1|                                 0|                               0|                               0|                                1|                             0|                            0|                                0|                            0|                              0|                            0|                           0|                             1|                              0|                             0|                            0|                                0|                            0|                              0|                            0|                           0|                             1|                              0|                  1|                    0|                 0|                  0|                 1|                  0|                 0|                1|                0|                   0|                     0|                  0|                0|               0|                  1|                    0|               0|                        1|                        0|                       0|                            0|                        0|                         0|                         0|                        0|                         0|                    0|                     0|                    0|                     1|                   0|                    0|                0|               0|                0|              0|              1|              0|              0|
|               0|                  0|                 1|               0|                  0|                 0|                    0|                    0|                  0|                   1|                0|               0|                   0|               1|                0|               0|                 0|              0|                0|                 0|                 0|            1|            0|           0|              0|           0|          0|           0|          1|             0|           0|                         0|                           0|                     1|                        0|                   0|                     1|                     0|                1|                 0|                 1|                 0|                0|                     0|                0|                 0|                  0|                0|                  0|               0|                 0|                  0|                      0|                     1|                   0|                0|               0|                 1|                       0|                  0|                   0|                                 0|                               0|                               0|                                1|                                 0|                               0|                               0|                                1|                             0|                            0|                                0|                            0|                              0|                            0|                           0|                             1|                              0|                             0|                            0|                                0|                            0|                              0|                            0|                           0|                             1|                              0|                  1|                    0|                 0|                  0|                 1|                  0|                 0|                1|                0|                   0|                     1|                  0|                0|               0|                  0|                    0|               0|                        0|                        1|                       0|                            0|                        0|                         0|                         0|                        0|                         0|                    1|                     0|                    0|                     0|                   0|                    0|                1|               0|                0|              0|              0|              0|              0|
|               0|                  0|                 1|               0|                  0|                 0|                    0|                    0|                  1|                   0|                0|               0|                   0|               0|                0|               0|                 0|              0|                0|                 1|                 1|            0|            1|           0|              0|           0|          0|           0|          0|             0|           0|                         0|                           0|                     1|                        0|                   1|                     0|                     0|                1|                 0|                 0|                 1|                0|                     0|                0|                 0|                  0|                0|                  0|               0|                 0|                  0|                      1|                     0|                   0|                1|               0|                 0|                       0|                  0|                   0|                                 0|                               0|                               0|                                1|                                 0|                               0|                               0|                                1|                             0|                            0|                                0|                            0|                              0|                            0|                           0|                             1|                              0|                             0|                            0|                                0|                            0|                              0|                            0|                           0|                             1|                              0|                  1|                    0|                 0|                  0|                 1|                  0|                 0|                1|                0|                   0|                     0|                  0|                0|               0|                  1|                    0|               0|                        1|                        0|                       0|                            0|                        0|                         0|                         0|                        0|                         0|                    0|                     0|                    1|                     0|                   0|                    0|                1|               0|                0|              0|              0|              0|              0|

<br />

Model Training and Prediction
=============================

As for the parameters, since the dataset is rather small, parameters are set as follows:

-   number of iterations, or number of trees `nrounds = 2`
-   maximum depth of tree, `max.depth = 2`
-   number of threads used, `nthread = 2`

The model is trained using xgboost as below:

``` r
#building the classifier using xgboost
model <- xgboost(data = train$data, label = train$label, nrounds = 2, objective = "binary:logistic", verbose = 1, max.depth = 2, nthread = 2, eta = 1)
```

    ## [0]  train-error:0.046522
    ## [1]  train-error:0.022263

``` r
#building the predictor
pred <- predict(model, test$data)

#preview of pred
head(pred)
```

    ## [1] 0.28583017 0.92392391 0.28583017 0.28583017 0.05169873 0.92392391

Tree Visualization
------------------

``` r
xgb.plot.tree(feature_names = agaricus.train$data@Dimnames[[2]], model = model)
```

<!--html_preserve-->

<script type="application/json" data-for="htmlwidget-bf3c25d4a7c112adc0e2">{"x":{"diagram":"graph LR;0-0(odor=none)-->|>= -1.00136e-05|0-2>spore-print-color=green];0-0(odor=none<br/>Cover: 1628.25<br/>Gain: 4000.53)-->|< -1.00136e-05|0-1>stalk-root=club];0-1(stalk-root=club)-->|>= -1.00136e-05|0-4>Leaf];0-1(stalk-root=club<br/>Cover: 924.5<br/>Gain: 1158.21)-->|< -1.00136e-05|0-3>Leaf];0-2(spore-print-color=green)-->|>= -1.00136e-05|0-6>Leaf];0-2(spore-print-color=green<br/>Cover: 703.75<br/>Gain: 198.174)-->|< -1.00136e-05|0-5>Leaf];1-0(stalk-root=rooted)-->|>= -1.00136e-05|1-2>Leaf];1-0(stalk-root=rooted<br/>Cover: 788.852<br/>Gain: 832.545)-->|< -1.00136e-05|1-1>odor=none];1-1(odor=none)-->|>= -1.00136e-05|1-4>Leaf];1-1(odor=none<br/>Cover: 768.39<br/>Gain: 569.725)-->|< -1.00136e-05|1-3>Leaf];classDef greenNode fill:#A2EB86, stroke:#04C4AB, stroke-width:2px;classDef redNode fill:#FFA070, stroke:#FF5E5E, stroke-width:2px;class 0-1,0-3,0-5,1-1,1-3 greenNode;class 0-2,0-4,0-6,1-2,1-4 redNode"},"evals":[],"jsHooks":[]}</script>
<!--/html_preserve-->
<br />

<br />

<br />

<br />

<br />

<br />

<br />

<br />

<br />

<br />

<br />

<br />

Transformation
--------------

Note that `pred` returns a vector of probabilities where if the probability is \> 0.5, it represents a predicted class label of 1, which means "poisonous"

``` r
#perform the transformation 
pred_t <- as.numeric(pred > 0.5)

head(pred_t)
```

    ## [1] 0 1 0 0 0 1

<br />

Model Evaluation
================

To evaluate the model, **Average Error** can be a simple evaluator alongside the **Confusion Matrix**.

Average Error
-------------

``` r
err <- mean(as.numeric(pred > 0.5) != test$label)
print(paste("test-error=", err))
```

    ## [1] "test-error= 0.0217256362507759"

Confusion Matrix
----------------

``` r
confusionMatrix(test$label, pred_t, dnn = c("actual", "predicted"), positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##       predicted
    ## actual   0   1
    ##      0 813  22
    ##      1  13 763
    ##                                           
    ##                Accuracy : 0.9783          
    ##                  95% CI : (0.9699, 0.9848)
    ##     No Information Rate : 0.5127          
    ##     P-Value [Acc > NIR] : <2e-16          
    ##                                           
    ##                   Kappa : 0.9565          
    ##  Mcnemar's Test P-Value : 0.1763          
    ##                                           
    ##             Sensitivity : 0.9720          
    ##             Specificity : 0.9843          
    ##          Pos Pred Value : 0.9832          
    ##          Neg Pred Value : 0.9737          
    ##              Prevalence : 0.4873          
    ##          Detection Rate : 0.4736          
    ##    Detection Prevalence : 0.4817          
    ##       Balanced Accuracy : 0.9781          
    ##                                           
    ##        'Positive' Class : 1               
    ## 

ROC Curve
---------

``` r
pred_obj <- prediction(test$label, pred_t )
perf <- performance(pred_obj, measure = "tpr", x.measure = "fpr")
plot(perf, main = "ROC Curve for Poisonous Mushrooms", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
```

![](Poisonous_Mushroom_Using_XGBOOST_files/figure-markdown_github/unnamed-chunk-10-1.png)

Feature Importance
------------------

To plot the feature importance, an `importance` matrix must first be constructed:

``` r
#constructing importance matrix
importance_matrix <- xgb.importance(feature_names = agaricus.train$data@Dimnames[[2]], model = model)

#plotting
xgb.plot.importance(importance_matrix)
```

![](Poisonous_Mushroom_Using_XGBOOST_files/figure-markdown_github/unnamed-chunk-11-1.png)

<br />

Improving the Model
===================

To improve the model, some parameters must be tuned.

Parameter Tuning
----------------

As we limited some of the parameters earlier in the model, this time, let's see if we remove the criteria of `maxdepth`, `eta`, and `nthread`.

``` r
model <- xgboost(data = train$data, label = train$label, nrounds = 2, objective = "binary:logistic")
```

    ## [0]  train-error:0.000614
    ## [1]  train-error:0.001228

``` r
pred <- predict(model, test$data)

pred_t <- ifelse(pred > 0.5, 1, 0)

confusionMatrix(test$label, pred_t, dnn = c("actual", "predicted"), positive = "1")
```

    ## Confusion Matrix and Statistics
    ## 
    ##       predicted
    ## actual   0   1
    ##      0 835   0
    ##      1   0 776
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9977, 1)
    ##     No Information Rate : 0.5183     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ##                                      
    ##             Sensitivity : 1.0000     
    ##             Specificity : 1.0000     
    ##          Pos Pred Value : 1.0000     
    ##          Neg Pred Value : 1.0000     
    ##              Prevalence : 0.4817     
    ##          Detection Rate : 0.4817     
    ##    Detection Prevalence : 0.4817     
    ##       Balanced Accuracy : 1.0000     
    ##                                      
    ##        'Positive' Class : 1          
    ##
