Titanic Survival Prediction
================
Jason Chan
October 15, 2016

-   Objective
-   Libraries Used
-   Data Overview
    -   Training Data
    -   Test Data
    -   Summary Statistic
    -   Survival Rate
    -   Gender Survival Rate
    -   Age & Missing Values
-   Handling Missing Values
-   Other Interesting Variables
-   Algo 1:Using Decision Trees
    -   Output
        -   Insights So Far
-   Feature Engineering
    -   Titles
    -   Family Size
    -   Family Surname
-   Newly Engineered Features
-   New Prediction
-   Algo 2: Random Forest
    -   Handling Missing Values
        -   Age
        -   Embarked
        -   Fare
    -   Reducing Levels
    -   Prediction
-   Conditional Inference Trees
-   Final Prediction
    -   Evaluation from Sampling

Objective
=========

The objective of this project is to predict the survival of Titanic passengers. The rescue operation was well known for saving women and children as the priority as safety boats were very limited. Hence, given a test set of Titanic passengers, who would make it and who would not ?

Libraries Used
==============

``` r
library(rpart) # Recursive Partitioning and Regression Trees
#For better visualization
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)
library(party) # conditional inference trees
```

Data Overview
=============

The training dataset consists of 12 variables and 891 entries. The variables provided were **Passenger id, Pclass, Name, Sex, Age, Sibsp, Parch, Ticket, Fare, Cabin, Embarked.** Our target variable is **Survived**.

### Training Data

A quick look at the training data:

    ##   PassengerId Survived Pclass
    ## 1           1        0      3
    ## 2           2        1      1
    ## 3           3        1      3
    ## 4           4        1      1
    ## 5           5        0      3
    ## 6           6        0      3
    ##                                                  Name    Sex Age SibSp
    ## 1                             Braund, Mr. Owen Harris   male  22     1
    ## 2 Cumings, Mrs. John Bradley (Florence Briggs Thayer) female  38     1
    ## 3                              Heikkinen, Miss. Laina female  26     0
    ## 4        Futrelle, Mrs. Jacques Heath (Lily May Peel) female  35     1
    ## 5                            Allen, Mr. William Henry   male  35     0
    ## 6                                    Moran, Mr. James   male  NA     0
    ##   Parch           Ticket    Fare Cabin Embarked
    ## 1     0        A/5 21171  7.2500              S
    ## 2     0         PC 17599 71.2833   C85        C
    ## 3     0 STON/O2. 3101282  7.9250              S
    ## 4     0           113803 53.1000  C123        S
    ## 5     0           373450  8.0500              S
    ## 6     0           330877  8.4583              Q

### Test Data

A quick look at the test data.

    ##   PassengerId Pclass                                         Name    Sex
    ## 1         892      3                             Kelly, Mr. James   male
    ## 2         893      3             Wilkes, Mrs. James (Ellen Needs) female
    ## 3         894      2                    Myles, Mr. Thomas Francis   male
    ## 4         895      3                             Wirz, Mr. Albert   male
    ## 5         896      3 Hirvonen, Mrs. Alexander (Helga E Lindqvist) female
    ## 6         897      3                   Svensson, Mr. Johan Cervin   male
    ##    Age SibSp Parch  Ticket    Fare Cabin Embarked
    ## 1 34.5     0     0  330911  7.8292              Q
    ## 2 47.0     1     0  363272  7.0000              S
    ## 3 62.0     0     0  240276  9.6875              Q
    ## 4 27.0     0     0  315154  8.6625              S
    ## 5 22.0     1     1 3101298 12.2875              S
    ## 6 14.0     0     0    7538  9.2250              S

### Summary Statistic

A summary statistic of our data:

    ##   PassengerId       Survived          Pclass     
    ##  Min.   :  1.0   Min.   :0.0000   Min.   :1.000  
    ##  1st Qu.:223.5   1st Qu.:0.0000   1st Qu.:2.000  
    ##  Median :446.0   Median :0.0000   Median :3.000  
    ##  Mean   :446.0   Mean   :0.3838   Mean   :2.309  
    ##  3rd Qu.:668.5   3rd Qu.:1.0000   3rd Qu.:3.000  
    ##  Max.   :891.0   Max.   :1.0000   Max.   :3.000  
    ##                                                  
    ##                                     Name         Sex           Age       
    ##  Abbing, Mr. Anthony                  :  1   female:314   Min.   : 0.42  
    ##  Abbott, Mr. Rossmore Edward          :  1   male  :577   1st Qu.:20.12  
    ##  Abbott, Mrs. Stanton (Rosa Hunt)     :  1                Median :28.00  
    ##  Abelson, Mr. Samuel                  :  1                Mean   :29.70  
    ##  Abelson, Mrs. Samuel (Hannah Wizosky):  1                3rd Qu.:38.00  
    ##  Adahl, Mr. Mauritz Nils Martin       :  1                Max.   :80.00  
    ##  (Other)                              :885                NA's   :177    
    ##      SibSp           Parch             Ticket         Fare       
    ##  Min.   :0.000   Min.   :0.0000   1601    :  7   Min.   :  0.00  
    ##  1st Qu.:0.000   1st Qu.:0.0000   347082  :  7   1st Qu.:  7.91  
    ##  Median :0.000   Median :0.0000   CA. 2343:  7   Median : 14.45  
    ##  Mean   :0.523   Mean   :0.3816   3101295 :  6   Mean   : 32.20  
    ##  3rd Qu.:1.000   3rd Qu.:0.0000   347088  :  6   3rd Qu.: 31.00  
    ##  Max.   :8.000   Max.   :6.0000   CA 2144 :  6   Max.   :512.33  
    ##                                   (Other) :852                   
    ##          Cabin     Embarked
    ##             :687    :  2   
    ##  B96 B98    :  4   C:168   
    ##  C23 C25 C27:  4   Q: 77   
    ##  G6         :  4   S:644   
    ##  C22 C26    :  3           
    ##  D          :  3           
    ##  (Other)    :186

### Survival Rate

We could already do a quick calculation of survival rate, 0 for dead and 1 for survived:

    ## 
    ##     0     1 
    ## 61.62 38.38

Chances of survival seems dim close to 2 in 5 people would survive the incident.

### Gender Survival Rate

How about survival rates of gender ?

    ##         
    ##             0    1
    ##   female 0.26 0.74
    ##   male   0.81 0.19

Seems like survival rate for females are high as majority of them survived. Whereas for males, survival is only at 19%. This is not surprising as women and children were prioritized in the rescue.

### Age & Missing Values

It seems like there are about 177 missing values in the **age** variable of our dataset.

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
    ##    0.42   20.12   28.00   29.70   38.00   80.00     177

Handling Missing Values
=======================

For now, the missing ages will be assumed as the mean, which falls under the adult category of late 20's. A separate variable **Child** is create to categorize the age as Child (below 18) and adult (above 20). Passengers with **NA** entry will have the value of 0 in the Child columnm, indicating adult.

The proportion of the aggregated survival rate for gender and age categorization is as follow:

    ##   Child    Sex  Survived
    ## 1     0 female 0.7528958
    ## 2     1 female 0.6909091
    ## 3     0   male 0.1657033
    ## 4     1   male 0.3965517

Still, most females survived, regardless of being child or adult. As for male, most still do not survive.

Other Interesting Variables
===========================

We shall look at how their **socio-economic classes** and **fare** paid relates to the survival rate.

    ##    Fare2 Pclass    Sex  Survived
    ## 1  20-30      1 female 0.8333333
    ## 2    30+      1 female 0.9772727
    ## 3  10-20      2 female 0.9142857
    ## 4  20-30      2 female 0.9000000
    ## 5    30+      2 female 1.0000000
    ## 6  10-20      3 female 0.5813953
    ## 7  20-30      3 female 0.3333333
    ## 8    30+      3 female 0.1250000
    ## 9    <10      3 female 0.5937500
    ## 10 20-30      1   male 0.4000000
    ## 11   30+      1   male 0.3837209
    ## 12   <10      1   male 0.0000000
    ## 13 10-20      2   male 0.1587302
    ## 14 20-30      2   male 0.1600000
    ## 15   30+      2   male 0.2142857
    ## 16   <10      2   male 0.0000000
    ## 17 10-20      3   male 0.2368421
    ## 18 20-30      3   male 0.1250000
    ## 19   30+      3   male 0.2400000
    ## 20   <10      3   male 0.1115385

It seems that 3rd class females who purchased a fare more than 20+ do not survive. Perhaps they were located nearest to the impact of the iceberg or maybe they were located furthest from the emergency exit ?

Based on the fact that 3rd class women who purchased fares about 20 do not survive, let's make a new prediction based on our test data:

Algo 1:Using Decision Trees
===========================

The algorithm used for this will be Recursive Partitioning and Regression Trees. The variables that will be used are **Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked**. The remaining variables **Passenger ID, Name, Ticket Number, and Cabin Number** are unique identifiers and are not useful in the model.

Output
------

![](book1_files/figure-markdown_github/unnamed-chunk-11-1.png)

From the RPART, we can see that for males ages less than 6.5, there is a good chance of survival although only a small portion of them consists of this group. This tallies completely with the Naval code of rescue and hence it is not surprising.

### Insights So Far

The following groups have higher chances of survival:

-   Female (regardless of child or adult)
-   Female with socio-economic class of less than 2.5
-   Female with fare more than 23
-   Male with age less than 6.5, representing male children

Feature Engineering
===================

In this section, more value will be attempted to be squeezed out from the data set by carrying out feature engineering.

Titles
------

Titles sich as **Sir, Mr, Madam, Countess** can be extracted from the name column of the dataset. The title usually comes after the first name. Below are the titles present in the dataset. The titles Mme and Mlle has been combined to Mlle. Capt, Don, Major, Sir is all replaced by Sir and Dona, Lady, The Countess, Jonkheer has been replaced by Lady.

    ## 
    ##    Col     Dr   Lady Master   Miss   Mlle     Mr    Mrs     Ms    Rev 
    ##      4      8      4     61    260      3    757    197      2      8 
    ##    Sir 
    ##      5

Now, who survives by title ?

    ##     Title  Survived
    ## 1     Col 0.5000000
    ## 2      Dr 0.4285714
    ## 3    Lady 0.6666667
    ## 4  Master 0.5750000
    ## 5    Miss 0.6978022
    ## 6    Mlle 1.0000000
    ## 7      Mr 0.1566731
    ## 8     Mrs 0.7920000
    ## 9      Ms 1.0000000
    ## 10    Rev 0.0000000
    ## 11    Sir 0.4000000

Family Size
-----------

Another engineered feature would be **Family Size**, where it is just the sum of **Parch, SibSp, and 1 (indicate self)**. This could provide more insights on whether family size could have affected survivability in any way. Perhaps, a large family size would have been searching high and low for the little ones.

    ##   FamilySize  Survived
    ## 1          1 0.3035382
    ## 2          2 0.5527950
    ## 3          3 0.5784314
    ## 4          4 0.7241379
    ## 5          5 0.2000000
    ## 6          6 0.1363636
    ## 7          7 0.3333333
    ## 8          8 0.0000000
    ## 9         11 0.0000000

Family Surname
--------------

Could one family be having more trouble than the other due to the size ? Perhaps sorting out the situation on the lifeboats ? Small families are family size of less than 2.

    ##    FamilyID Survived
    ## 1    11Sage      0.0
    ## 2   3Abbott      0.5
    ## 3 3Appleton      1.0
    ## 4 3Beckwith      1.0
    ## 5   3Boulos      0.0
    ## 6   3Bourke      0.0

    ##      FamilyID  Survived
    ## 82  6Richards 1.0000000
    ## 83     6Skoog 0.0000000
    ## 84 7Andersson 0.1250000
    ## 85   7Asplund 0.7500000
    ## 86   8Goodwin 0.0000000
    ## 87      Small 0.3610315

Newly Engineered Features
=========================

The newly engineered features by far are columns **Surname & FamilyID**. This is the preview of the training dataset:

    ##   PassengerId Pclass                                                Name
    ## 1           1      3                             Braund, Mr. Owen Harris
    ## 2           2      1 Cumings, Mrs. John Bradley (Florence Briggs Thayer)
    ## 3           3      3                              Heikkinen, Miss. Laina
    ## 4           4      1        Futrelle, Mrs. Jacques Heath (Lily May Peel)
    ## 5           5      3                            Allen, Mr. William Henry
    ## 6           6      3                                    Moran, Mr. James
    ##      Sex Age SibSp Parch           Ticket    Fare Cabin Embarked Survived
    ## 1   male  22     1     0        A/5 21171  7.2500              S        0
    ## 2 female  38     1     0         PC 17599 71.2833   C85        C        1
    ## 3 female  26     0     0 STON/O2. 3101282  7.9250              S        1
    ## 4 female  35     1     0           113803 53.1000  C123        S        1
    ## 5   male  35     0     0           373450  8.0500              S        0
    ## 6   male  NA     0     0           330877  8.4583              Q        0
    ##   Title FamilySize   Surname FamilyID
    ## 1    Mr          2    Braund    Small
    ## 2   Mrs          2   Cumings    Small
    ## 3  Miss          1 Heikkinen    Small
    ## 4   Mrs          2  Futrelle    Small
    ## 5    Mr          1     Allen    Small
    ## 6    Mr          1     Moran    Small

New Prediction
==============

The new prediction using RPART now also includes the newly engineered features of **Surname & FamilyID**. ![](book1_files/figure-markdown_github/unnamed-chunk-19-1.png)

Algo 2: Random Forest
=====================

Applying the Random Forest algorithm will reduce the overfitting as the many trees grown will cancel out the effect on average. However, the setback is that Random Forest can't deal with missing (NA) values like RPART which uses surrogate values.

Handling Missing Values
-----------------------

The features having missing values are **Age, Embarked, and Fare**. The summary of the features are:

``` r
summary(combi$Age)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
    ##    0.17   21.00   28.00   29.88   39.00   80.00     263

``` r
summary(combi$Embarked)
```

    ##       C   Q   S 
    ##   2 270 123 914

``` r
summary(combi$Fare)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
    ##   0.000   7.896  14.450  33.300  31.280 512.300       1

### Age

For the missing values (NA) of **Age**, a regression with the non NA Age values is used against the other features to estimate the age.

``` r
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize, data=combi[!is.na(combi$Age),], method="anova")

combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age), ])
```

### Embarked

For missing values (NA) in **Embarked**, there are only two missing values and their values has been assigned to "S" as it makes up 70% of the dataset.

``` r
which(combi$Embarked == "")
```

    ## [1]  62 830

``` r
combi$Embarked[c(62, 830)] <- "S"
combi$Embarked <- factor(combi$Embarked)
```

### Fare

Since there is only 1 NA in **Fare**, it will be assigned to the median value.

``` r
which(is.na(combi$Fare))
```

    ## [1] 1044

``` r
combi$Fare[1044] <- median(combi$Fare, na.rm = TRUE)
```

Reducing Levels
---------------

Note that Random Forest can only digest factors up to 32 levels whereas **FamilyID** has double the amount. The levels will be reduced by increasing the cutoff size of a small family size to 3 instead of 2.

``` r
combi$FamilyID2 <- combi$FamilyID
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
combi$FamilyID2 <- factor(combi$FamilyID2)
# reduced to 22 levels
```

Prediction
----------

Before applying the random forest, we force **Survived** to be a factor so that we force the model to predict the classification of the target variable to two levels.

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##    0.17   22.00   28.86   29.70   36.50   80.00

    ##   PassengerId       Pclass          Name               Sex     
    ##  Min.   :   1   Min.   :1.000   Length:1309        female:466  
    ##  1st Qu.: 328   1st Qu.:2.000   Class :character   male  :843  
    ##  Median : 655   Median :3.000   Mode  :character               
    ##  Mean   : 655   Mean   :2.295                                  
    ##  3rd Qu.: 982   3rd Qu.:3.000                                  
    ##  Max.   :1309   Max.   :3.000                                  
    ##                                                                
    ##       Age            SibSp            Parch            Ticket    
    ##  Min.   : 0.17   Min.   :0.0000   Min.   :0.000   CA. 2343:  11  
    ##  1st Qu.:22.00   1st Qu.:0.0000   1st Qu.:0.000   1601    :   8  
    ##  Median :28.86   Median :0.0000   Median :0.000   CA 2144 :   8  
    ##  Mean   :29.70   Mean   :0.4989   Mean   :0.385   3101295 :   7  
    ##  3rd Qu.:36.50   3rd Qu.:1.0000   3rd Qu.:0.000   347077  :   7  
    ##  Max.   :80.00   Max.   :8.0000   Max.   :9.000   347082  :   7  
    ##                                                   (Other) :1261  
    ##       Fare                     Cabin      Embarked    Survived     
    ##  Min.   :  0.000                  :1014   C:270    Min.   :0.0000  
    ##  1st Qu.:  7.896   C23 C25 C27    :   6   Q:123    1st Qu.:0.0000  
    ##  Median : 14.454   B57 B59 B63 B66:   5   S:916    Median :0.0000  
    ##  Mean   : 33.281   G6             :   5            Mean   :0.3838  
    ##  3rd Qu.: 31.275   B96 B98        :   4            3rd Qu.:1.0000  
    ##  Max.   :512.329   C22 C26        :   4            Max.   :1.0000  
    ##                    (Other)        : 271            NA's   :418     
    ##      Title       FamilySize       Surname                FamilyID   
    ##  Mr     :757   Min.   : 1.000   Length:1309        Small     :1074  
    ##  Miss   :260   1st Qu.: 1.000   Class :character   11Sage    :  11  
    ##  Mrs    :197   Median : 1.000   Mode  :character   7Andersson:   9  
    ##  Master : 61   Mean   : 1.884                      8Goodwin  :   8  
    ##  Dr     :  8   3rd Qu.: 2.000                      7Asplund  :   7  
    ##  Rev    :  8   Max.   :11.000                      6Fortune  :   6  
    ##  (Other): 18                                       (Other)   : 194  
    ##       FamilyID2   
    ##  Small     :1194  
    ##  11Sage    :  11  
    ##  7Andersson:   9  
    ##  8Goodwin  :   8  
    ##  7Asplund  :   7  
    ##  6Fortune  :   6  
    ##  (Other)   :  74

    ##   C   Q   S 
    ## 270 123 916

    ## integer(0)

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   0.000   7.896  14.450  33.280  31.280 512.300

    ## integer(0)

``` r
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2, data=train, importance=TRUE, ntree=2000)
set.seed(415)
varImpPlot(fit, main = "Which Variables Were Important ?")
```

![](book1_files/figure-markdown_github/unnamed-chunk-26-1.png)

It clearly seems that the engineered features, **Title, FamilyID** were quite important and doing well. **Title** is the top for both measures.

Conditional Inference Trees
===========================

An attempt to improve Random Forest is by using Conditional Inference Trees where it handles more factor levels than Random Forest and uses statistical tests in decision making.

``` r
set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, data = train, controls=cforest_unbiased(ntree=2000, mtry=3))
Prediction <- predict(fit, test, OOB=TRUE, type = "response")
```

Final Prediction
================

After constructing the CForest, this is the final prediction for the test set:

``` r
Prediction <- predict(fit, test, OOB=TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Name = test$Name, Survived = Prediction)
head(submit)
```

    ##   PassengerId                                         Name Survived
    ## 1         892                             Kelly, Mr. James        0
    ## 2         893             Wilkes, Mrs. James (Ellen Needs)        0
    ## 3         894                    Myles, Mr. Thomas Francis        0
    ## 4         895                             Wirz, Mr. Albert        0
    ## 5         896 Hirvonen, Mrs. Alexander (Helga E Lindqvist)        1
    ## 6         897                   Svensson, Mr. Johan Cervin        0

Evaluation from Sampling
------------------------

Sample from the traning set.
