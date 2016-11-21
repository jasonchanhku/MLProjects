Performing OCR with SVM
================
Jason Chan
November 21, 2016

-   Libraries Used
-   Objective
-   Step 1: Data Exploration
    -   Data Preview
    -   Data Structure
    -   Features
-   Step 2: Data Preparation
-   Step 3: Model Training
    -   Interpretation
-   Step 4: Model Evaluation
    -   Accuracy
-   Step 5: Improving the Model
    -   Improved Model Evaluation
-   Conclusion

For the fully functional html version, please visit <http://www.rpubs.com/jasonchanhku/ocr>

Libraries Used
==============

``` r
library(knitr) #kable
library(kernlab) #ksvm (kernlab)
library(caret) #confusion matrix
```

<br />

Objective
=========

This project aims to develop a model similar to those used at the core of the **Optical Character Recognition (OCR)** using **Support Vector Machines**. Paper-based documents can be processed by converting printed or handwritten text into an electronic form to be saved in a database.

<br />

Step 1: Data Exploration
========================

The data is obtained from the UCI Machine Learning Data Repository <http://archive.ics.uci.edu/ml>. The data set contains 20,000 examples of alphabets with its statistical attributes after scanned and converted into pixels.

Data Preview
------------

A preview of the dataset is enabled using `head()` and `kable()`.

``` r
#Load the dataset
letters <- read.csv(file = "Machine-Learning-with-R-datasets-master/letterdata.csv")

#Data preview
kable(head(letters), caption = "Partial Data Preview")
```

| letter |  xbox|  ybox|  width|  height|  onpix|  xbar|  ybar|  x2bar|  y2bar|  xybar|  x2ybar|  xy2bar|  xedge|  xedgey|  yedge|  yedgex|
|:-------|-----:|-----:|------:|-------:|------:|-----:|-----:|------:|------:|------:|-------:|-------:|------:|-------:|------:|-------:|
| T      |     2|     8|      3|       5|      1|     8|    13|      0|      6|      6|      10|       8|      0|       8|      0|       8|
| I      |     5|    12|      3|       7|      2|    10|     5|      5|      4|     13|       3|       9|      2|       8|      4|      10|
| D      |     4|    11|      6|       8|      6|    10|     6|      2|      6|     10|       3|       7|      3|       7|      3|       9|
| N      |     7|    11|      6|       6|      3|     5|     9|      4|      6|      4|       4|      10|      6|      10|      2|       8|
| G      |     2|     1|      3|       1|      1|     8|     6|      6|      6|      6|       5|       9|      1|       7|      5|      10|
| S      |     4|    11|      5|       8|      3|     8|     8|      6|      9|      5|       6|       6|      0|       8|      9|       7|

Data Structure
--------------

The structure of the data is obtained using `str()`:

``` r
str(letters)
```

    ## 'data.frame':    20000 obs. of  17 variables:
    ##  $ letter: Factor w/ 26 levels "A","B","C","D",..: 20 9 4 14 7 19 2 1 10 13 ...
    ##  $ xbox  : int  2 5 4 7 2 4 4 1 2 11 ...
    ##  $ ybox  : int  8 12 11 11 1 11 2 1 2 15 ...
    ##  $ width : int  3 3 6 6 3 5 5 3 4 13 ...
    ##  $ height: int  5 7 8 6 1 8 4 2 4 9 ...
    ##  $ onpix : int  1 2 6 3 1 3 4 1 2 7 ...
    ##  $ xbar  : int  8 10 10 5 8 8 8 8 10 13 ...
    ##  $ ybar  : int  13 5 6 9 6 8 7 2 6 2 ...
    ##  $ x2bar : int  0 5 2 4 6 6 6 2 2 6 ...
    ##  $ y2bar : int  6 4 6 6 6 9 6 2 6 2 ...
    ##  $ xybar : int  6 13 10 4 6 5 7 8 12 12 ...
    ##  $ x2ybar: int  10 3 3 4 5 6 6 2 4 1 ...
    ##  $ xy2bar: int  8 9 7 10 9 6 6 8 8 9 ...
    ##  $ xedge : int  0 2 3 6 1 0 2 1 1 8 ...
    ##  $ xedgey: int  8 8 7 10 7 8 8 6 6 1 ...
    ##  $ yedge : int  0 4 3 2 5 9 7 2 1 1 ...
    ##  $ yedgex: int  8 10 9 8 10 7 10 7 7 8 ...

Features
--------

The target variable and features used are identified as the following:

**Target Variable**

-   letter

**Features Used**

-   attributes measure such characteristics as the horizontal and vertical dimensions of the glyph, the proportion of black (versus white) pixels, and the average horizontal and vertical position of the pixels.

<br />

Step 2: Data Preparation
========================

Another feature requirement for SVM besides being numeric is that they must be **small scaled** and needs to be normalized. However, the SVM model in the R package does the normalization and hence, the data can be prepared right away. The proportion used for training and test data is 80% training and 20% test.

``` r
#preparing training set
letters_train <- letters[1:16000, ]

#preparing test set
letters_test <- letters[16001:20000, ]
```

<br />

Step 3: Model Training
======================

The SVM model that will be trained is from the `kernlab` package using `ksvm()`. The advantage of this package is that it can be used with `caret` and be evaluated and trained automatically. `ksvm()` also uses the **Gaussian GBF** by default.

``` r
#building the classifier using a dot product as the kernel function
letters_m <- ksvm(letter ~ . , data = letters_train, kernel = "vanilladot")
```

    ##  Setting default kernel parameters

``` r
#print basic information of the model
letters_m
```

    ## Support Vector Machine object of class "ksvm" 
    ## 
    ## SV type: C-svc  (classification) 
    ##  parameter : cost C = 1 
    ## 
    ## Linear (vanilla) kernel function. 
    ## 
    ## Number of Support Vectors : 7037 
    ## 
    ## Objective Function Value : -14.1746 -20.0072 -23.5628 -6.2009 -7.5524 -32.7694 -49.9786 -18.1824 -62.1111 -32.7284 -16.2209 -32.2837 -28.9777 -51.2195 -13.276 -35.6217 -30.8612 -16.5256 -14.6811 -32.7475 -30.3219 -7.7956 -11.8138 -32.3463 -13.1262 -9.2692 -153.1654 -52.9678 -76.7744 -119.2067 -165.4437 -54.6237 -41.9809 -67.2688 -25.1959 -27.6371 -26.4102 -35.5583 -41.2597 -122.164 -187.9178 -222.0856 -21.4765 -10.3752 -56.3684 -12.2277 -49.4899 -9.3372 -19.2092 -11.1776 -100.2186 -29.1397 -238.0516 -77.1985 -8.3339 -4.5308 -139.8534 -80.8854 -20.3642 -13.0245 -82.5151 -14.5032 -26.7509 -18.5713 -23.9511 -27.3034 -53.2731 -11.4773 -5.12 -13.9504 -4.4982 -3.5755 -8.4914 -40.9716 -49.8182 -190.0269 -43.8594 -44.8667 -45.2596 -13.5561 -17.7664 -87.4105 -107.1056 -37.0245 -30.7133 -112.3218 -32.9619 -27.2971 -35.5836 -17.8586 -5.1391 -43.4094 -7.7843 -16.6785 -58.5103 -159.9936 -49.0782 -37.8426 -32.8002 -74.5249 -133.3423 -11.1638 -5.3575 -12.438 -30.9907 -141.6924 -54.2953 -179.0114 -99.8896 -10.288 -15.1553 -3.7815 -67.6123 -7.696 -88.9304 -47.6448 -94.3718 -70.2733 -71.5057 -21.7854 -12.7657 -7.4383 -23.502 -13.1055 -239.9708 -30.4193 -25.2113 -136.2795 -140.9565 -9.8122 -34.4584 -6.3039 -60.8421 -66.5793 -27.2816 -214.3225 -34.7796 -16.7631 -135.7821 -160.6279 -45.2949 -25.1023 -144.9059 -82.2352 -327.7154 -142.0613 -158.8821 -32.2181 -32.8887 -52.9641 -25.4937 -47.9936 -6.8991 -9.7293 -36.436 -70.3907 -187.7611 -46.9371 -89.8103 -143.4213 -624.3645 -119.2204 -145.4435 -327.7748 -33.3255 -64.0607 -145.4831 -116.5903 -36.2977 -66.3762 -44.8248 -7.5088 -217.9246 -12.9699 -30.504 -2.0369 -6.126 -14.4448 -21.6337 -57.3084 -20.6915 -184.3625 -20.1052 -4.1484 -4.5344 -0.828 -121.4411 -7.9486 -58.5604 -21.4878 -13.5476 -5.646 -15.629 -28.9576 -20.5959 -76.7111 -27.0119 -94.7101 -15.1713 -10.0222 -7.6394 -1.5784 -87.6952 -6.2239 -99.3711 -101.0906 -45.6639 -24.0725 -61.7702 -24.1583 -52.2368 -234.3264 -39.9749 -48.8556 -34.1464 -20.9664 -11.4525 -123.0277 -6.4903 -5.1865 -8.8016 -9.4618 -21.7742 -24.2361 -123.3984 -31.4404 -88.3901 -30.0924 -13.8198 -9.2701 -3.0823 -87.9624 -6.3845 -13.968 -65.0702 -105.523 -13.7403 -13.7625 -50.4223 -2.933 -8.4289 -80.3381 -36.4147 -112.7485 -4.1711 -7.8989 -1.2676 -90.8037 -21.4919 -7.2235 -47.9557 -3.383 -20.433 -64.6138 -45.5781 -56.1309 -6.1345 -18.6307 -2.374 -72.2553 -111.1885 -106.7664 -23.1323 -19.3765 -54.9819 -34.2953 -64.4756 -20.4115 -6.689 -4.378 -59.141 -34.2468 -58.1509 -33.8665 -10.6902 -53.1387 -13.7478 -20.1987 -55.0923 -3.8058 -60.0382 -235.4841 -12.6837 -11.7407 -17.3058 -9.7167 -65.8498 -17.1051 -42.8131 -53.1054 -25.0437 -15.302 -44.0749 -16.9582 -62.9773 -5.204 -5.2963 -86.1704 -3.7209 -6.3445 -1.1264 -122.5771 -23.9041 -355.0145 -31.1013 -32.619 -4.9664 -84.1048 -134.5957 -72.8371 -23.9002 -35.3077 -11.7119 -22.2889 -1.8598 -59.2174 -8.8994 -150.742 -1.8533 -1.9711 -9.9676 -0.5207 -26.9229 -30.429 -5.6289 
    ## Training error : 0.130062

Interpretation
--------------

Based on the constructed classifier, the following can be noted from the result:

-   **Cost Value (C):** The higher the C, the harder the model will perform separation, the lower, the more generalized it is

-   **Linear kernel function:** does not transform, just mapping. Other options include "rbfdot", "polydot", "tanhdot"

-   **Training Error:** The lower the training error, the better it will perform on the test set

<br />

Step 4: Model Evaluation
========================

The model is now ready to be evaluated using the test dataset.

``` r
#by default, parameter = "response" for predicted class, if type = "probabilities", then it returns probabilities  
letters_p <- predict(letters_m, letters_test, type = "response")

#preview of predicted target variable
head(letters_p)
```

    ## [1] U N V X N H
    ## Levels: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z

Accuracy
--------

One could use a `confusionMatrix()` from `caret` but since there are 26 class levels, it would be too messy to interpret. Hence, the following is done:

``` r
# Create a TRUE FALSE vector list
agreement <- letters_p == letters_test$letter

#Get the proportion
prop.table(table(agreement))
```

    ## agreement
    ##   FALSE    TRUE 
    ## 0.16075 0.83925

The SVM model gives an accuracy rate of 84%, which is not bad, but can certainly be improved.

<br />

Step 5: Improving the Model
===========================

The current model uses a **linear kernel function**. By using a more **complex** kernel function, the data and features can be mapped into a **higer dimension** to fit the data better.

A very popular convention is to begin with the **Gaussian RBF kernel**, which performs well with many types of data. The **Cost Value (C)** shall also be set to 2 to make the model try harder to seperate the class levels.

It shall be trained as follows:

``` r
#building the new classifier
letters_m2 <- ksvm(letter ~. , data = letters_train, kernel = "rbfdot", C = 2)

#building the predictor
letter_p2 <- predict(letters_m2, letters_test)
```

Improved Model Evaluation
-------------------------

``` r
agreement2 <- letter_p2 == letters_test$letter

prop.table(table(agreement2))
```

    ## agreement2
    ##   FALSE    TRUE 
    ## 0.05275 0.94725

Conclusion
==========

Just by using a **more complex kernel function**, the accuracy of the model improved from 84% to 94%. If this level of performance is still unsatisfactory for the OCR program, other kernels could be tested, or the cost of constraints parameter C could be varied to modify the width of the decision boundary.
