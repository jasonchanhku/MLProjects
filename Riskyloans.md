Identifying Risky Bank Loans with C5.0
================
Jason Chan
November 14, 2016

-   Libraries Used
-   Objective
-   Step 1: Data Exploration
    -   Data Preview
    -   Useful Tables
        -   Checking Balance
        -   Savings Balance
        -   Loan Duration
        -   Loan Amount
        -   Default Applicants
    -   Data Visualization
        -   Age
        -   Loan Amount
        -   Loan Duration
        -   Job
        -   Checking Balance
        -   Savings Balance
        -   Purpose
    -   Preliminary Insights
-   Step 2: Data Preparation
    -   Random Sample
-   Step 3: Model Training
    -   Explanation
-   Step 4: Evaluating Model Performance
-   Step 5: Improving the Model
    -   Adaptive Boosting

For complete html functions, please visit www.rpubs.com/jasonchanhku/loans

Libraries Used
==============

``` r
library(fBasics) #summary statistics
library(plotly) #data visulization
library(dplyr) #count function
library(C50) #C50 Decision Tree algorithm
library(gmodels) #CrossTable()
library(caret) #Confusion Matrix
```

<br />

Objective
=========

The global financial crisis of 2007-2008 highlighted the importance of transparency and rigor in banking practices. As the availability of credit was limited, banks tightened their lending systems and turned to machine learning to more accurately identify risky loans.

Therefore, objective of this project is to develop a simple credit approval model using the **C5.0 decision tree algorithm**.

The target variable of interest is **default status**.

<br />

Step 1: Data Exploration
========================

The data is obtained from donated to the UCI Machine Learning Data Repository (<http://archive.ics.uci.edu/ml>) by Hans Hofmann of the University of Hamburg. The dataset contains information on loans obtained from a credit agency in Germany.

The credit dataset includes 1,000 examples on loans, plus a set of numeric and nominal features indicating the characteristics of the loan and the loan applicant. A class variable indicates whether the loan went into default.

``` r
credit <- read.csv(file = "Machine-Learning-with-R-datasets-master/credit.csv")
credit$default[credit$default == 1] <- "no"
credit$default[credit$default == 2] <- "yes"
credit$default <- as.factor(credit$default)
```

Data Preview
------------

Below is a partial table preview of the dataset:

``` r
knitr::kable(head(credit), caption = "Credit Information Dataset")
```

| checking\_balance |  months\_loan\_duration| credit\_history | purpose   |  amount| savings\_balance | employment\_length |  installment\_rate| personal\_status | other\_debtors |  residence\_history| property                 |  age| installment\_plan | housing  |  existing\_credits| job                |  dependents| telephone | foreign\_worker | default |
|:------------------|-----------------------:|:----------------|:----------|-------:|:-----------------|:-------------------|------------------:|:-----------------|:---------------|-------------------:|:-------------------------|----:|:------------------|:---------|------------------:|:-------------------|-----------:|:----------|:----------------|:--------|
| \< 0 DM           |                       6| critical        | radio/tv  |    1169| unknown          | \> 7 yrs           |                  4| single male      | none           |                   4| real estate              |   67| none              | own      |                  2| skilled employee   |           1| yes       | yes             | no      |
| 1 - 200 DM        |                      48| repaid          | radio/tv  |    5951| \< 100 DM        | 1 - 4 yrs          |                  2| female           | none           |                   2| real estate              |   22| none              | own      |                  1| skilled employee   |           1| none      | yes             | yes     |
| unknown           |                      12| critical        | education |    2096| \< 100 DM        | 4 - 7 yrs          |                  2| single male      | none           |                   3| real estate              |   49| none              | own      |                  1| unskilled resident |           2| none      | yes             | no      |
| \< 0 DM           |                      42| repaid          | furniture |    7882| \< 100 DM        | 4 - 7 yrs          |                  2| single male      | guarantor      |                   4| building society savings |   45| none              | for free |                  1| skilled employee   |           2| none      | yes             | no      |
| \< 0 DM           |                      24| delayed         | car (new) |    4870| \< 100 DM        | 1 - 4 yrs          |                  3| single male      | none           |                   4| unknown/none             |   53| none              | for free |                  2| skilled employee   |           2| none      | yes             | yes     |
| unknown           |                      36| repaid          | education |    9055| unknown          | 1 - 4 yrs          |                  2| single male      | none           |                   4| unknown/none             |   35| none              | for free |                  1| unskilled resident |           2| yes       | yes             | no      |

The general structure of the dataset:

``` r
str(credit)
```

    ## 'data.frame':    1000 obs. of  21 variables:
    ##  $ checking_balance    : Factor w/ 4 levels "1 - 200 DM","< 0 DM",..: 2 1 4 2 2 4 4 1 4 1 ...
    ##  $ months_loan_duration: int  6 48 12 42 24 36 24 36 12 30 ...
    ##  $ credit_history      : Factor w/ 5 levels "critical","delayed",..: 1 5 1 5 2 5 5 5 5 1 ...
    ##  $ purpose             : Factor w/ 10 levels "business","car (new)",..: 8 8 5 6 2 5 6 3 8 2 ...
    ##  $ amount              : int  1169 5951 2096 7882 4870 9055 2835 6948 3059 5234 ...
    ##  $ savings_balance     : Factor w/ 5 levels "101 - 500 DM",..: 5 3 3 3 3 5 2 3 4 3 ...
    ##  $ employment_length   : Factor w/ 5 levels "0 - 1 yrs","1 - 4 yrs",..: 4 2 3 3 2 2 4 2 3 5 ...
    ##  $ installment_rate    : int  4 2 2 2 3 2 3 2 2 4 ...
    ##  $ personal_status     : Factor w/ 4 levels "divorced male",..: 4 2 4 4 4 4 4 4 1 3 ...
    ##  $ other_debtors       : Factor w/ 3 levels "co-applicant",..: 3 3 3 2 3 3 3 3 3 3 ...
    ##  $ residence_history   : int  4 2 3 4 4 4 4 2 4 2 ...
    ##  $ property            : Factor w/ 4 levels "building society savings",..: 3 3 3 1 4 4 1 2 3 2 ...
    ##  $ age                 : int  67 22 49 45 53 35 53 35 61 28 ...
    ##  $ installment_plan    : Factor w/ 3 levels "bank","none",..: 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ housing             : Factor w/ 3 levels "for free","own",..: 2 2 2 1 1 1 2 3 2 2 ...
    ##  $ existing_credits    : int  2 1 1 1 2 1 1 1 1 2 ...
    ##  $ job                 : Factor w/ 4 levels "mangement self-employed",..: 2 2 4 2 2 4 2 1 4 1 ...
    ##  $ dependents          : int  1 1 2 2 2 2 1 1 1 1 ...
    ##  $ telephone           : Factor w/ 2 levels "none","yes": 2 1 1 1 1 2 1 2 1 1 ...
    ##  $ foreign_worker      : Factor w/ 2 levels "no","yes": 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ default             : Factor w/ 2 levels "no","yes": 1 2 1 1 2 1 1 1 1 2 ...

Useful Tables
-------------

Below are some useful tables that offers summary statistics from the loan features such as **checking, savings, duration, amount, and default**.

### Checking Balance

``` r
knitr::kable(data.frame(table(credit$checking_balance)), Caption = "Checking Balance", col.names = c("Checking Balance", "Frequency") )
```

| Checking Balance |  Frequency|
|:-----------------|----------:|
| 1 - 200 DM       |        269|
| \< 0 DM          |        274|
| \> 200 DM        |         63|
| unknown          |        394|

### Savings Balance

``` r
knitr::kable(data.frame(table(credit$savings_balance)), Caption = "Savings Balance", col.names = c("Savings Balance", "Frequency"))
```

| Savings Balance |  Frequency|
|:----------------|----------:|
| 101 - 500 DM    |        103|
| 501 - 1000 DM   |         63|
| \< 100 DM       |        603|
| \> 1000 DM      |         48|
| unknown         |        183|

### Loan Duration

``` r
knitr::kable(basicStats(credit$months_loan_duration), Caption = "Loan Duration", digits = 2, col.names = c("Value"))
```

|             |     Value|
|-------------|---------:|
| nobs        |   1000.00|
| NAs         |      0.00|
| Minimum     |      4.00|
| Maximum     |     72.00|
| 1. Quartile |     12.00|
| 3. Quartile |     24.00|
| Mean        |     20.90|
| Median      |     18.00|
| Sum         |  20903.00|
| SE Mean     |      0.38|
| LCL Mean    |     20.15|
| UCL Mean    |     21.65|
| Variance    |    145.42|
| Stdev       |     12.06|
| Skewness    |      1.09|
| Kurtosis    |      0.90|

### Loan Amount

``` r
knitr::kable(basicStats(credit$amount), Caption = "Loan Amount", digits = 2, col.names = c("Amount"))
```

|             |      Amount|
|-------------|-----------:|
| nobs        |     1000.00|
| NAs         |        0.00|
| Minimum     |      250.00|
| Maximum     |    18424.00|
| 1. Quartile |     1365.50|
| 3. Quartile |     3972.25|
| Mean        |     3271.26|
| Median      |     2319.50|
| Sum         |  3271258.00|
| SE Mean     |       89.26|
| LCL Mean    |     3096.09|
| UCL Mean    |     3446.42|
| Variance    |  7967843.47|
| Stdev       |     2822.74|
| Skewness    |        1.94|
| Kurtosis    |        4.25|

### Default Applicants

``` r
knitr::kable(data.frame(table(credit$default)), col.names = c("Default", "Amount"))
```

| Default |  Amount|
|:--------|-------:|
| no      |     700|
| yes     |     300|

Data Visualization
------------------

The data visualization will focus on identifying patterns of **key features** which clearly distinguishes an applicant's **default status**.

### Age

``` r
plot_ly(credit, y = ~age, color = ~default, type = "box")
```

<!--html_preserve-->

<script type="application/json" data-for="htmlwidget-a2a316890664e600ca8a">{"x":{"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"yaxis":{"domain":[0,1],"title":"age"},"xaxis":{"domain":[0,1]}},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"modeBarButtonsToRemove":["sendDataToCloud"]},"data":[{"y":[67,49,45,35,53,35,61,22,28,53,25,31,48,44,48,44,26,36,39,42,34,36,27,30,57,33,31,37,24,30,26,44,24,35,39,23,39,28,29,30,25,31,26,31,23,27,50,26,48,29,22,25,30,46,51,41,66,51,39,22,47,24,58,52,29,27,30,56,54,20,54,61,34,36,36,41,24,24,35,26,32,30,35,31,23,28,35,47,27,36,41,24,63,30,40,34,24,27,47,21,38,27,35,44,27,30,27,23,30,39,51,28,46,42,38,24,29,36,48,45,38,34,36,30,36,70,36,32,20,25,26,33,42,52,31,65,50,31,68,33,29,28,36,52,27,26,38,38,43,26,21,55,33,45,51,39,31,23,24,64,26,23,30,32,30,27,22,51,35,25,42,35,39,51,27,35,25,52,35,26,39,46,35,24,27,35,23,57,27,55,36,57,32,36,38,25,32,37,36,32,26,49,29,23,50,49,63,37,35,26,31,49,26,44,56,46,20,45,43,32,54,49,33,24,22,40,25,26,29,38,48,32,27,34,28,36,39,49,34,31,28,75,23,28,31,24,26,25,33,37,43,23,23,34,23,38,46,49,28,61,37,36,21,36,27,22,40,36,33,23,63,34,36,52,39,25,26,26,25,21,40,27,27,30,19,39,31,31,32,55,46,43,39,28,27,43,43,27,26,20,35,40,35,23,31,20,30,47,34,21,29,46,20,74,36,33,25,23,37,65,39,30,29,41,35,55,30,29,34,35,29,36,27,32,37,36,34,38,34,63,32,26,35,36,24,25,39,44,23,26,57,30,44,52,62,35,26,26,27,38,39,40,32,28,42,49,36,28,45,32,26,20,54,37,40,43,36,44,23,26,30,31,42,41,32,41,26,25,75,37,45,60,61,37,32,35,23,45,27,67,49,29,37,23,34,41,38,26,22,27,24,27,33,27,49,26,52,36,21,58,42,36,32,45,23,22,74,33,45,29,22,48,27,37,49,27,22,35,41,36,64,28,23,47,28,21,34,38,33,32,32,50,35,22,37,28,41,23,50,35,50,27,34,27,47,27,31,42,24,26,33,64,26,56,37,33,47,31,34,27,30,35,31,25,25,29,44,28,50,29,38,24,40,47,41,32,35,25,37,32,46,25,63,40,32,31,31,66,41,47,36,33,44,28,37,29,35,45,32,23,41,22,30,28,23,26,33,49,23,25,74,31,59,24,27,40,31,28,63,26,36,52,66,37,25,38,67,60,31,60,35,40,38,41,27,51,32,22,22,54,35,54,48,24,35,24,26,65,55,26,28,54,62,24,43,27,24,47,35,30,38,44,42,21,23,63,46,28,50,47,35,28,59,43,35,45,33,40,28,26,27,32,20,27,42,37,24,40,46,26,24,29,40,36,28,36,38,48,36,65,34,34,40,43,46,38,34,29,31,28,35,33,44,42,40,36,20,24,27,46,33,34,25,28,32,28,37,30,21,58,43,24,30,42,23,30,46,45,31,31,42,46,30,38,40,29,57,49,37,30,30,47,29,22,26,54,29,40,22,43,33,57,64,42,28,30,25,33,64,29,48,37,34,23,30,50,31,40,38,27],"type":"box","name":"no","line":{"fillcolor":"rgba(102,194,165,0.5)","color":"rgba(102,194,165,1)"},"xaxis":"x","yaxis":"y"},{"y":[22,53,28,25,24,60,32,44,63,25,37,58,57,52,23,61,25,37,40,34,44,47,28,33,58,39,39,25,30,23,25,27,30,29,29,66,22,20,33,31,33,34,26,53,28,30,40,36,74,20,54,34,36,21,34,27,40,21,50,66,27,53,22,26,30,23,61,29,24,22,24,29,37,45,28,34,32,48,28,26,42,37,44,33,24,25,31,28,32,30,24,24,23,44,24,32,29,28,23,26,23,25,42,60,37,57,38,46,27,22,28,42,35,33,33,25,55,29,25,26,41,30,34,61,31,35,29,22,23,28,33,26,47,42,20,29,27,38,24,27,34,26,23,24,53,31,28,33,42,23,31,34,43,24,34,22,28,29,27,31,24,37,36,31,23,27,30,33,20,47,60,20,40,32,23,36,31,30,34,28,50,22,48,22,21,32,38,65,29,44,19,25,26,27,40,27,26,38,40,37,45,42,41,23,43,41,24,29,46,24,25,35,27,34,24,24,21,25,59,21,23,26,37,23,55,32,39,35,24,30,31,25,25,25,23,50,27,39,51,24,26,24,54,46,26,41,33,36,47,23,29,25,48,29,23,68,57,33,32,29,28,35,25,27,43,53,23,42,43,25,31,32,68,33,39,22,55,46,39,22,30,28,42,30,43,31,24,28,26,45,35,23,29,36,47,25,49,33,26,23],"type":"box","name":"yes","line":{"fillcolor":"rgba(141,160,203,0.5)","color":"rgba(141,160,203,1)"},"xaxis":"x","yaxis":"y"}],"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":[]}</script>
<!--/html_preserve-->
As visualized, default applicants are of **younger age**. Perhaps the younger age have less awareness about financial planning.

### Loan Amount

``` r
plot_ly(credit, y = ~amount, color = ~default, type = "box")
```

<!--html_preserve-->

<script type="application/json" data-for="htmlwidget-b4400baa9f93b53989b9">{"x":{"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"yaxis":{"domain":[0,1],"title":"amount"},"xaxis":{"domain":[0,1]}},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"modeBarButtonsToRemove":["sendDataToCloud"]},"data":[{"y":[1169,2096,7882,9055,2835,6948,3059,1567,1403,2424,8072,3430,2134,2647,2241,1804,2069,1374,426,409,2415,1913,4020,5866,1264,1474,6110,1225,458,2333,1158,6204,6187,1393,2299,1352,7228,2073,2333,5965,1262,3378,783,9566,1961,1391,1537,3181,5190,2171,1007,2394,8133,730,1164,5954,1526,4771,9436,3832,1213,1568,1755,2315,1412,1295,2249,618,1409,3617,1318,2012,2622,2337,7057,1469,2323,932,1919,2445,6078,7721,1410,1449,392,6260,1680,3578,2132,2366,3868,1768,781,2121,701,1860,8487,2708,1984,10144,1240,8613,2728,1881,709,4795,3416,2288,3566,860,682,5371,1582,1346,1924,5848,7758,6967,1288,339,3512,1898,2872,1055,1262,7308,909,2978,1577,3972,763,1414,2577,338,1963,571,3777,1360,1175,3244,2108,1382,2760,936,1168,5117,1495,10623,1935,1424,6568,1413,3074,3835,3342,932,3104,3913,3021,1364,625,1200,707,2978,4657,2613,3149,2507,2141,866,1544,1291,2522,1595,1185,3447,1258,717,1204,1925,666,2251,4151,2030,7418,2684,3812,1154,1657,1603,5302,2748,1231,6304,1533,999,2662,1402,12169,2697,2404,1262,1901,3368,1574,1445,1520,3878,10722,4788,7582,1092,1076,6419,4796,7629,4675,1287,2515,2745,672,1038,1543,4811,276,5381,5511,3749,1494,708,4351,701,3643,2910,2659,1028,3398,5801,1525,4473,1068,6615,2101,4169,1521,5743,3599,3213,4439,3949,1459,882,3758,1743,1236,3229,727,2331,776,1239,3399,2247,1766,1542,3850,3650,3001,3079,6070,2146,13756,2320,846,362,2212,1283,1330,4272,2238,1126,7374,2326,1449,1820,983,3249,1957,2406,11760,2578,2348,1516,1473,1887,802,2899,2197,1053,3235,1967,7253,1597,5842,8471,2782,3186,2028,958,1591,2779,2743,1149,1313,3448,1872,2058,2136,660,1287,3394,1884,1620,2629,1244,2576,1424,11054,518,2759,2670,2679,3905,343,4594,3620,1721,3017,754,1950,2924,7238,2764,4679,1238,2569,5152,1037,1478,3573,1201,3622,960,1163,3077,3757,1418,3518,1934,1237,368,2122,1585,1301,1323,5493,1126,2360,1413,8588,4686,2687,585,2255,609,1361,1203,700,5507,3488,1113,7966,1532,1503,662,2273,1503,1311,3105,1374,3612,3049,2032,1255,2022,1552,8858,996,1750,1995,1199,2964,683,4712,1553,2578,3979,5954,5433,806,1082,2788,2930,1927,937,3124,1388,2133,2799,1289,385,1965,1572,3863,2329,1275,2828,4526,2051,1300,3357,3632,12204,9157,3676,640,3652,1858,1979,2116,4042,3832,3660,1444,1393,1376,15653,1493,1308,1851,1880,4583,7476,2346,3973,10222,4221,6361,1297,2241,1050,1047,6314,3496,3609,3017,4139,5742,10366,2080,4530,5150,2384,1453,1538,2279,1478,5103,9857,6527,1347,2862,2753,3651,975,2631,2896,4716,2284,1236,1103,926,1800,1905,1377,2503,2528,5324,1206,2118,629,2476,1138,7596,3077,1505,3148,6148,790,250,1316,1275,6403,760,2603,3380,3990,4380,6761,2325,1048,3160,2483,14179,1797,2511,5248,3029,428,841,5771,1299,1393,5045,2214,2463,1155,2901,3617,1655,2812,3275,2223,1480,3535,3509,5711,3872,1940,1410,6468,1941,2675,2751,6313,1221,2892,3062,2301,1258,717,1549,1597,1795,4272,976,7472,590,930,9283,907,484,7432,1338,1554,15857,1101,3016,731,3780,1602,6681,2375,5084,886,601,2957,2611,2993,1559,3422,3976,1249,1364,4042,1471,10875,1474,894,3343,3577,5804,4526,2221,2389,3331,7409,652,7678,1343,1382,874,3590,1322,1940,3595,6742,7814,9277,2181,1098,2825,6614,7824,2442,1829,2171,5800,1169,8947,2606,1592,2186,3485,10477,1386,1278,1107,3763,3711,3594,3195,4454,2991,2142,2848,1817,12749,2002,1049,1867,1344,1747,1224,522,1498,2063,6842,3527,1546,929,1455,1845,8358,2859,3590,1893,1231,3656,1154,3069,1740,2353,3556,454,1715,3568,7166,3939,1514,7393,2831,1258,753,2427,2923,2028,1433,6289,1409,6579,1743,3565,1569,1936,3959,2390,1736,3857,804,4576],"type":"box","name":"no","line":{"fillcolor":"rgba(102,194,165,0.5)","color":"rgba(102,194,165,1)"},"xaxis":"x","yaxis":"y"},{"y":[5951,4870,5234,1295,4308,1199,1282,12579,6836,4746,2100,6143,2225,6468,6229,1953,14421,1819,1977,3965,5943,12612,1108,797,15945,11938,6458,7855,7174,4281,1835,1924,639,3499,6887,766,2462,1282,1131,1935,950,2064,3414,7485,9572,4455,1647,884,5129,674,4591,3844,3915,3031,1501,951,4297,902,5293,1908,10961,7865,1478,4210,1823,14555,2767,915,4605,433,2150,2149,802,8978,3060,11998,4611,1024,9398,9960,3804,1344,10127,727,1237,685,2746,4249,1938,1864,7408,11590,4110,3384,1275,1136,959,6199,1246,4463,2406,2473,3446,14782,7685,14318,12976,1223,8648,2039,939,2292,1381,2579,1042,2762,1190,11328,1484,609,719,5096,1842,1512,4817,3386,1659,3092,448,654,1245,3114,1209,8318,2996,9034,3123,1216,1207,1309,6850,759,7127,3190,7119,2302,2631,2319,7763,1534,6350,2864,1333,626,6999,1331,2278,5003,3552,1928,1546,12389,1372,6758,3234,2820,1056,2384,2039,1217,2246,2718,1358,931,1442,4241,2775,918,1837,3349,2671,741,1240,1808,3441,1530,3914,2600,1437,1553,1980,1355,4370,750,4623,7980,1386,947,684,1922,2303,8086,888,900,4843,2580,5595,1123,6331,6560,2969,1198,14027,1337,433,1228,2570,1882,6416,1987,11560,4280,1274,976,1555,1285,1271,691,2124,12680,3108,8065,1371,4933,836,6224,5998,1188,7511,9271,1778,9629,3051,3931,1345,2712,3966,4165,8335,1216,11816,2327,1082,5179,1943,6761,709,2235,1442,3959,2169,2439,2210,1422,4057,795,15672,4153,2625,4736,3161,18424,14896,2359,3345,1366,6872,697,10297,1670,1919,745,6288,3349,1533,3621,2145,4113,10974,4006,2397,2520,1193,7297,2538,1264,8386,4844,8229,1845],"type":"box","name":"yes","line":{"fillcolor":"rgba(141,160,203,0.5)","color":"rgba(141,160,203,1)"},"xaxis":"x","yaxis":"y"}],"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":[]}</script>
<!--/html_preserve-->
As visualized, default applicants tend to have **higher loan amounts**. Risky applicants tend to loan huge amounts perhaps due to underestimation.

### Loan Duration

``` r
plot_ly(credit, y=~months_loan_duration, color = ~default, type = "box")
```

<!--html_preserve-->

<script type="application/json" data-for="htmlwidget-6a6b84657ff9b8b2ad54">{"x":{"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"yaxis":{"domain":[0,1],"title":"months_loan_duration"},"xaxis":{"domain":[0,1]}},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"modeBarButtonsToRemove":["sendDataToCloud"]},"data":[{"y":[6,12,42,36,24,36,12,12,15,24,30,24,9,6,10,12,10,6,6,12,7,18,24,18,12,12,48,10,9,30,12,18,30,11,36,6,11,12,24,27,12,18,6,36,18,9,15,24,27,12,12,36,36,7,8,42,12,11,54,30,15,18,24,10,12,18,18,12,12,24,12,12,18,36,20,24,36,6,9,12,12,24,14,6,15,18,12,48,10,12,24,12,10,12,12,12,48,15,18,60,12,27,15,12,6,36,27,21,48,6,12,36,18,6,10,36,24,24,9,12,24,6,24,18,15,10,36,6,11,24,12,8,12,6,12,21,24,15,16,18,6,6,24,9,12,27,12,30,12,12,24,12,9,36,36,6,18,36,24,10,12,12,12,24,15,36,24,9,12,18,4,12,30,6,12,12,24,12,6,24,6,12,24,9,60,24,15,11,12,24,18,12,10,36,24,24,18,12,48,9,18,12,24,15,12,18,15,24,47,48,48,12,12,24,42,48,12,10,18,21,6,10,6,30,9,48,24,24,4,12,24,12,15,24,18,18,8,12,24,36,6,24,13,24,10,24,21,18,18,10,15,13,24,6,9,18,10,12,12,18,12,12,6,12,18,18,18,36,18,10,60,18,7,6,20,22,12,30,18,18,18,15,9,18,12,36,6,9,39,12,36,24,18,18,14,18,24,15,24,24,33,10,36,18,21,15,12,12,21,18,28,18,9,5,6,24,9,6,24,42,12,12,20,9,7,12,36,6,12,24,24,11,6,18,36,15,12,12,18,24,48,33,24,6,39,24,12,15,12,24,30,15,12,12,24,10,6,12,6,6,12,24,18,6,36,9,15,24,39,36,15,12,24,6,6,6,6,24,24,18,26,15,4,6,36,12,24,24,6,18,18,24,12,24,24,48,12,6,12,9,24,6,24,24,24,48,30,24,15,9,15,12,24,24,12,9,12,9,12,12,24,21,24,7,10,24,24,18,15,21,24,48,60,6,12,21,12,15,6,42,9,24,15,12,24,60,12,15,24,18,30,48,24,14,48,30,18,12,21,6,6,24,30,48,30,24,36,60,6,30,24,24,18,6,12,15,24,36,60,10,36,9,12,15,15,24,6,24,6,12,12,18,15,24,30,27,15,9,9,18,21,9,30,30,18,24,20,9,6,15,24,24,8,24,4,36,18,6,24,10,21,24,39,13,15,21,15,6,12,30,6,24,15,12,24,12,10,12,12,24,21,24,12,36,18,36,18,24,12,20,18,22,48,24,6,24,24,9,12,24,9,24,18,20,12,12,6,12,42,8,6,36,6,6,36,12,12,8,18,21,48,24,24,12,4,24,24,21,24,18,21,24,9,24,15,36,24,10,15,9,24,27,15,18,12,36,12,36,6,24,15,12,11,18,36,30,24,24,30,18,24,36,28,27,15,12,36,18,36,21,12,15,20,36,15,24,12,21,36,15,9,36,30,11,10,18,48,12,18,30,12,24,9,12,12,6,24,12,10,24,4,15,48,12,18,12,24,30,9,24,6,21,15,6,30,15,42,11,15,24,30,24,6,18,21,24,15,42,13,24,24,12,15,18,36,12,12,30,12,45],"type":"box","name":"no","line":{"fillcolor":"rgba(102,194,165,0.5)","color":"rgba(102,194,165,1)"},"xaxis":"x","yaxis":"y"},{"y":[48,24,30,12,48,24,24,24,60,45,18,48,36,12,36,36,48,36,36,42,24,36,12,12,54,24,18,36,42,33,21,18,12,12,36,12,18,12,18,24,15,24,21,30,36,36,21,18,9,12,24,48,27,45,9,12,18,12,27,30,48,12,9,36,24,6,21,24,48,18,30,12,15,14,48,30,6,24,36,48,36,24,48,12,8,12,36,30,24,18,60,48,24,6,15,9,9,12,24,36,30,18,36,60,48,36,18,12,24,24,12,12,24,12,18,12,18,24,12,12,12,48,36,15,24,12,24,24,6,9,18,18,6,27,24,36,24,24,24,10,15,12,36,18,48,36,15,21,48,12,30,18,24,12,48,12,18,21,24,18,24,36,12,48,24,36,18,36,18,18,12,24,24,6,24,24,18,9,24,36,36,12,10,18,30,18,48,18,9,18,9,24,42,18,15,36,12,24,12,12,24,36,12,12,12,21,72,12,48,48,12,6,60,9,6,12,27,18,48,24,24,30,12,18,12,24,15,12,18,21,30,36,24,39,12,48,40,21,18,36,15,36,48,48,18,36,18,18,36,18,45,15,12,36,18,18,12,20,18,15,18,24,10,9,24,12,48,18,16,24,24,48,6,24,24,9,24,12,48,9,30,9,60,24,18,24,36,24,36,28,24,27,24,60,24,15,30,48,36,45],"type":"box","name":"yes","line":{"fillcolor":"rgba(141,160,203,0.5)","color":"rgba(141,160,203,1)"},"xaxis":"x","yaxis":"y"}],"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":[]}</script>
<!--/html_preserve-->
As visualized, default applicants tend to have **longer loan durations** perhaps due to their cash flows.

### Job

``` r
credit %>% count(default, job) %>% plot_ly(x = ~default, y = ~n, color = ~job, type = "bar")
```

<!--html_preserve-->

<script type="application/json" data-for="htmlwidget-d56e9920360019c8bf8f">{"x":{"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"xaxis":{"domain":[0,1],"title":"default","type":"category","categoryorder":"array","categoryarray":["no","yes"]},"yaxis":{"domain":[0,1],"title":"n"}},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"modeBarButtonsToRemove":["sendDataToCloud"]},"data":[{"x":["no","yes"],"y":[97,51],"type":"bar","name":"mangement self-employed","marker":{"fillcolor":"rgba(102,194,165,0.5)","color":"rgba(102,194,165,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[444,186],"type":"bar","name":"skilled employee","marker":{"fillcolor":"rgba(252,141,98,0.5)","color":"rgba(252,141,98,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[15,7],"type":"bar","name":"unemployed non-resident","marker":{"fillcolor":"rgba(141,160,203,0.5)","color":"rgba(141,160,203,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[144,56],"type":"bar","name":"unskilled resident","marker":{"fillcolor":"rgba(231,138,195,0.5)","color":"rgba(231,138,195,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"}],"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":[]}</script>
<!--/html_preserve-->
As visualized, defaulted applicants are **less skilled**. Perhaps this reflects their financial planning awareness.

### Checking Balance

``` r
credit %>% count(default, checking_balance) %>% plot_ly(x = ~default, y = ~n, color = ~checking_balance, type = "bar")
```

<!--html_preserve-->

<script type="application/json" data-for="htmlwidget-09dbe1760cad54887bc8">{"x":{"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"xaxis":{"domain":[0,1],"title":"default","type":"category","categoryorder":"array","categoryarray":["no","yes"]},"yaxis":{"domain":[0,1],"title":"n"}},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"modeBarButtonsToRemove":["sendDataToCloud"]},"data":[{"x":["no","yes"],"y":[164,105],"type":"bar","name":"1 - 200 DM","marker":{"fillcolor":"rgba(102,194,165,0.5)","color":"rgba(102,194,165,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[139,135],"type":"bar","name":"< 0 DM","marker":{"fillcolor":"rgba(252,141,98,0.5)","color":"rgba(252,141,98,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[49,14],"type":"bar","name":"> 200 DM","marker":{"fillcolor":"rgba(141,160,203,0.5)","color":"rgba(141,160,203,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[348,46],"type":"bar","name":"unknown","marker":{"fillcolor":"rgba(231,138,195,0.5)","color":"rgba(231,138,195,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"}],"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":[]}</script>
<!--/html_preserve-->
As visualized, defaulted applicants have **lower checking balance**.

### Savings Balance

``` r
credit %>% count(default, savings_balance) %>% plot_ly(x = ~default, y = ~n, color = ~savings_balance, type = "bar")
```

<!--html_preserve-->

<script type="application/json" data-for="htmlwidget-912900bd82bda7bd0bd9">{"x":{"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"xaxis":{"domain":[0,1],"title":"default","type":"category","categoryorder":"array","categoryarray":["no","yes"]},"yaxis":{"domain":[0,1],"title":"n"}},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"modeBarButtonsToRemove":["sendDataToCloud"]},"data":[{"x":["no","yes"],"y":[69,34],"type":"bar","name":"101 - 500 DM","marker":{"fillcolor":"rgba(102,194,165,0.5)","color":"rgba(102,194,165,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[52,11],"type":"bar","name":"501 - 1000 DM","marker":{"fillcolor":"rgba(252,141,98,0.5)","color":"rgba(252,141,98,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[386,217],"type":"bar","name":"< 100 DM","marker":{"fillcolor":"rgba(141,160,203,0.5)","color":"rgba(141,160,203,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[42,6],"type":"bar","name":"> 1000 DM","marker":{"fillcolor":"rgba(231,138,195,0.5)","color":"rgba(231,138,195,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[151,32],"type":"bar","name":"unknown","marker":{"fillcolor":"rgba(166,216,84,0.5)","color":"rgba(166,216,84,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"}],"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":[]}</script>
<!--/html_preserve-->
As visualized, defaulted applicants have **lower savings balance**.

### Purpose

``` r
credit %>% count(default, purpose) %>% plot_ly(x = ~default, y = ~n, color = ~purpose, type = "bar")
```

<!--html_preserve-->

<script type="application/json" data-for="htmlwidget-17825d16e406c21dfb43">{"x":{"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"xaxis":{"domain":[0,1],"title":"default","type":"category","categoryorder":"array","categoryarray":["no","yes"]},"yaxis":{"domain":[0,1],"title":"n"}},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"modeBarButtonsToRemove":["sendDataToCloud"]},"data":[{"x":["no","yes"],"y":[63,34],"type":"bar","name":"business","marker":{"fillcolor":"rgba(102,194,165,0.5)","color":"rgba(102,194,165,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[145,89],"type":"bar","name":"car (new)","marker":{"fillcolor":"rgba(228,156,113,0.5)","color":"rgba(228,156,113,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[86,17],"type":"bar","name":"car (used)","marker":{"fillcolor":"rgba(201,153,157,0.5)","color":"rgba(201,153,157,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[8,4],"type":"bar","name":"domestic appliances","marker":{"fillcolor":"rgba(175,154,200,0.5)","color":"rgba(175,154,200,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[28,22],"type":"bar","name":"education","marker":{"fillcolor":"rgba(226,148,184,0.5)","color":"rgba(226,148,184,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[123,58],"type":"bar","name":"furniture","marker":{"fillcolor":"rgba(176,209,99,0.5)","color":"rgba(176,209,99,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[7,5],"type":"bar","name":"others","marker":{"fillcolor":"rgba(227,217,62,0.5)","color":"rgba(227,217,62,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[218,62],"type":"bar","name":"radio/tv","marker":{"fillcolor":"rgba(245,207,100,0.5)","color":"rgba(245,207,100,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[14,8],"type":"bar","name":"repairs","marker":{"fillcolor":"rgba(219,192,155,0.5)","color":"rgba(219,192,155,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"},{"x":["no","yes"],"y":[8,1],"type":"bar","name":"retraining","marker":{"fillcolor":"rgba(179,179,179,0.5)","color":"rgba(179,179,179,1)","line":{"color":"transparent"}},"xaxis":"x","yaxis":"y"}],"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":[]}</script>
<!--/html_preserve-->
As visualized, of the defaulted applicants, most of them take loans to purchase **new cars**.

Preliminary Insights
--------------------

From the data preview and data visualization, we identify the following characteristics that distinguishes an **applicant who is likely to default**:

-   Low checking & savings balance
-   Less skilled
-   Younger age
-   Longer loan duration
-   Higher loan amounts
-   Apply loans to purchase new cars

<br />

Step 2: Data Preparation
========================

The training and test datasets will be split into a portion of 90% to 10%, leaving 900 values for training and 100 for test. However, as the dataset is not sorted in random order, this could cause **bias** if for example the data is sorted by loan amounts ascending. The model will train on small loans and test on big loans. Hence, **random sampling** is required.

Random Sample
-------------

Random sampling is applied as following (please click show code):

``` r
# To ensure results can be reproduced
set.seed(123)

# Sample 900 integers randomly from 1000
train_sample <- sample(1000,900)

#Subset randomly into training and test
credit_train <- credit[train_sample ,]
credit_test <- credit[-train_sample , ]
```

Let's check if both the train and test set have rather even split of the class levels. This is to prevent training bias.

``` r
knitr::kable(data.frame(prop.table(table(credit_train$default))), caption = "Training Set Proportion", col.names = c("Default Status", "Proportion"), digits = 3)
```

| Default Status |  Proportion|
|:---------------|-----------:|
| no             |       0.703|
| yes            |       0.297|

``` r
knitr::kable(data.frame(prop.table(table(credit_test$default))), caption = "Test Set Proportion", col.names = c("Default Status", "Proportion"), digits = 4)
```

| Default Status |  Proportion|
|:---------------|-----------:|
| no             |        0.67|
| yes            |        0.33|

Step 3: Model Training
======================

Since we have all the data we need, the C5.0 model can now be trained on the training set. To use the training set, the class labels must be removed from the training set and plugged into separately as labels.

The summary of the model is as follows:

``` r
#building the classifier
credit_model <- C5.0(credit_train[-21], credit_train$default)

credit_model
```

    ## 
    ## Call:
    ## C5.0.default(x = credit_train[-21], y = credit_train$default)
    ## 
    ## Classification Tree
    ## Number of samples: 900 
    ## Number of predictors: 20 
    ## 
    ## Tree size: 54 
    ## 
    ## Non-standard options: attempt to group attributes

``` r
summary(credit_model)
```

    ## 
    ## Call:
    ## C5.0.default(x = credit_train[-21], y = credit_train$default)
    ## 
    ## 
    ## C5.0 [Release 2.07 GPL Edition]      Tue Nov 15 14:57:35 2016
    ## -------------------------------
    ## 
    ## Class specified by attribute `outcome'
    ## 
    ## Read 900 cases (21 attributes) from undefined.data
    ## 
    ## Decision tree:
    ## 
    ## checking_balance in {> 200 DM,unknown}: no (412/50)
    ## checking_balance in {1 - 200 DM,< 0 DM}:
    ## :...other_debtors = guarantor:
    ##     :...months_loan_duration > 36: yes (4/1)
    ##     :   months_loan_duration <= 36:
    ##     :   :...installment_plan in {none,stores}: no (24)
    ##     :       installment_plan = bank:
    ##     :       :...purpose = car (new): yes (3)
    ##     :           purpose in {business,car (used),domestic appliances,education,
    ##     :                       furniture,others,radio/tv,repairs,
    ##     :                       retraining}: no (7/1)
    ##     other_debtors in {co-applicant,none}:
    ##     :...credit_history = critical: no (102/30)
    ##         credit_history = fully repaid: yes (27/6)
    ##         credit_history = fully repaid this bank:
    ##         :...other_debtors = co-applicant: no (2)
    ##         :   other_debtors = none: yes (26/8)
    ##         credit_history in {delayed,repaid}:
    ##         :...savings_balance in {501 - 1000 DM,> 1000 DM}: no (19/3)
    ##             savings_balance = 101 - 500 DM:
    ##             :...other_debtors = co-applicant: yes (3)
    ##             :   other_debtors = none:
    ##             :   :...personal_status in {divorced male,
    ##             :       :                   married male}: yes (6/1)
    ##             :       personal_status = female:
    ##             :       :...installment_rate <= 3: no (4/1)
    ##             :       :   installment_rate > 3: yes (4)
    ##             :       personal_status = single male:
    ##             :       :...age <= 41: no (15/2)
    ##             :           age > 41: yes (2)
    ##             savings_balance = unknown:
    ##             :...credit_history = delayed: no (8)
    ##             :   credit_history = repaid:
    ##             :   :...foreign_worker = no: no (2)
    ##             :       foreign_worker = yes:
    ##             :       :...checking_balance = < 0 DM:
    ##             :           :...telephone = none: yes (11/2)
    ##             :           :   telephone = yes:
    ##             :           :   :...amount <= 5045: no (5/1)
    ##             :           :       amount > 5045: yes (2)
    ##             :           checking_balance = 1 - 200 DM:
    ##             :           :...residence_history > 3: no (9)
    ##             :               residence_history <= 3: [S1]
    ##             savings_balance = < 100 DM:
    ##             :...months_loan_duration > 39:
    ##                 :...residence_history <= 1: no (2)
    ##                 :   residence_history > 1: yes (19/1)
    ##                 months_loan_duration <= 39:
    ##                 :...purpose in {car (new),retraining}: yes (47/16)
    ##                     purpose in {domestic appliances,others}: no (3)
    ##                     purpose = car (used):
    ##                     :...amount <= 8086: no (9/1)
    ##                     :   amount > 8086: yes (5)
    ##                     purpose = education:
    ##                     :...checking_balance = 1 - 200 DM: no (2)
    ##                     :   checking_balance = < 0 DM: yes (5)
    ##                     purpose = repairs:
    ##                     :...residence_history <= 3: yes (4/1)
    ##                     :   residence_history > 3: no (3)
    ##                     purpose = business:
    ##                     :...credit_history = delayed: yes (2)
    ##                     :   credit_history = repaid:
    ##                     :   :...age <= 34: no (5)
    ##                     :       age > 34: yes (2)
    ##                     purpose = radio/tv:
    ##                     :...employment_length in {0 - 1 yrs,
    ##                     :   :                     unemployed}: yes (14/5)
    ##                     :   employment_length = 4 - 7 yrs: no (3)
    ##                     :   employment_length = > 7 yrs:
    ##                     :   :...amount <= 932: yes (2)
    ##                     :   :   amount > 932: no (7)
    ##                     :   employment_length = 1 - 4 yrs:
    ##                     :   :...months_loan_duration <= 15: no (6)
    ##                     :       months_loan_duration > 15:
    ##                     :       :...amount <= 3275: yes (7)
    ##                     :           amount > 3275: no (2)
    ##                     purpose = furniture:
    ##                     :...residence_history <= 1: no (8/1)
    ##                         residence_history > 1:
    ##                         :...installment_plan in {bank,stores}: no (3/1)
    ##                             installment_plan = none:
    ##                             :...telephone = yes: yes (7/1)
    ##                                 telephone = none:
    ##                                 :...months_loan_duration > 27: yes (3)
    ##                                     months_loan_duration <= 27: [S2]
    ## 
    ## SubTree [S1]
    ## 
    ## property in {building society savings,unknown/none}: yes (4)
    ## property = other: no (6)
    ## property = real estate:
    ## :...job = skilled employee: yes (2)
    ##     job in {mangement self-employed,unemployed non-resident,
    ##             unskilled resident}: no (2)
    ## 
    ## SubTree [S2]
    ## 
    ## checking_balance = 1 - 200 DM: yes (5/2)
    ## checking_balance = < 0 DM:
    ## :...property in {building society savings,real estate,unknown/none}: no (8)
    ##     property = other:
    ##     :...installment_rate <= 1: no (2)
    ##         installment_rate > 1: yes (4)
    ## 
    ## 
    ## Evaluation on training data (900 cases):
    ## 
    ##      Decision Tree   
    ##    ----------------  
    ##    Size      Errors  
    ## 
    ##      54  135(15.0%)   <<
    ## 
    ## 
    ##     (a)   (b)    <-classified as
    ##    ----  ----
    ##     589    44    (a): class no
    ##      91   176    (b): class yes
    ## 
    ## 
    ##  Attribute usage:
    ## 
    ##  100.00% checking_balance
    ##   54.22% other_debtors
    ##   50.00% credit_history
    ##   32.56% savings_balance
    ##   25.22% months_loan_duration
    ##   19.78% purpose
    ##   10.11% residence_history
    ##    7.33% installment_plan
    ##    5.22% telephone
    ##    4.78% foreign_worker
    ##    4.56% employment_length
    ##    4.33% amount
    ##    3.44% personal_status
    ##    3.11% property
    ##    2.67% age
    ##    1.56% installment_rate
    ##    0.44% job
    ## 
    ## 
    ## Time: 0.0 secs

Explanation
-----------

First few lines of the summary reads the following:

-   if checking balance is more than 200 or unknown, less likely to default
-   412/50 indicates 412 classified correctly and 50 classified incorrectly
-   15% error rate is the sum of false positive and negative

<br />

Step 4: Evaluating Model Performance
====================================

Now it is time to put our test data to the test after training our model. After that, an evaluation is done to check its accuracy rate.

``` r
#making prediction
credit_pred <- predict(credit_model, credit_test)

#Evaluation
confusionMatrix(credit_pred, credit_test$default, dnn = c("Predicted", "Actual"))
```

    ## Confusion Matrix and Statistics
    ## 
    ##          Actual
    ## Predicted no yes
    ##       no  60  19
    ##       yes  7  14
    ##                                           
    ##                Accuracy : 0.74            
    ##                  95% CI : (0.6427, 0.8226)
    ##     No Information Rate : 0.67            
    ##     P-Value [Acc > NIR] : 0.08146         
    ##                                           
    ##                   Kappa : 0.3523          
    ##  Mcnemar's Test P-Value : 0.03098         
    ##                                           
    ##             Sensitivity : 0.8955          
    ##             Specificity : 0.4242          
    ##          Pos Pred Value : 0.7595          
    ##          Neg Pred Value : 0.6667          
    ##              Prevalence : 0.6700          
    ##          Detection Rate : 0.6000          
    ##    Detection Prevalence : 0.7900          
    ##       Balanced Accuracy : 0.6599          
    ##                                           
    ##        'Positive' Class : no              
    ## 

Turns out that the **accuracy is only 74%** and there is definitely room for improvement.

<br />

Step 5: Improving the Model
===========================

There are a couple of simple ways to adjust the C5.0 algorithm that may help to improve the performance of the model, both overall and for the more costly type of mistakes.

Adaptive Boosting
-----------------

Boosting is rooted in the notion that by **combining a number of weak performing learners**, you can create a team that is much **stronger** than any of the learners alone.

To add bossting to the model, an addtional parameter **trials** is added. It acts as an upper limit and stop adding trees if accuracy does not improve. The de facto standard is **10**.

The evaluation is as follows:

``` r
#making the classifier
credit_boost10 <- C5.0(credit_train[-21], credit_train$default, trials = 10)

#making the prediction
credit_boost_pred10 <- predict(credit_boost10, credit_test)

#Evaluation
confusionMatrix(credit_boost_pred10, credit_test$default)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction no yes
    ##        no  60  17
    ##        yes  7  16
    ##                                           
    ##                Accuracy : 0.76            
    ##                  95% CI : (0.6643, 0.8398)
    ##     No Information Rate : 0.67            
    ##     P-Value [Acc > NIR] : 0.03281         
    ##                                           
    ##                   Kappa : 0.4121          
    ##  Mcnemar's Test P-Value : 0.06619         
    ##                                           
    ##             Sensitivity : 0.8955          
    ##             Specificity : 0.4848          
    ##          Pos Pred Value : 0.7792          
    ##          Neg Pred Value : 0.6957          
    ##              Prevalence : 0.6700          
    ##          Detection Rate : 0.6000          
    ##    Detection Prevalence : 0.7700          
    ##       Balanced Accuracy : 0.6902          
    ##                                           
    ##        'Positive' Class : no              
    ## 

After adaptive boosting, the accuracy of the model **improved** to 76%.
