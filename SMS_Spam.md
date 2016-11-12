Filtering SMS Spam Using Naive Bayes
================
Jason Chan
November 11, 2016

-   Libraries Used
-   Objective
-   Step 1: Data Exploration
    -   Data Preview
    -   Features
        -   Target Variable
        -   Other Features
    -   Data Visualization
        -   Spam vs Ham Word Cloud
-   Step 2: Data Preprocessing
    -   Data Cleasing
        -   Corpus Creation
        -   Corpus Cleaning
        -   Post Corpus Cleansing
    -   Data Preparation
        -   Corpus Word Cloud
        -   Creating Training and Test Dataset
        -   Creating Indicator Features
-   Step 3: Model Training
-   Step 4: Model Evaluation
    -   CrossTable
    -   confusionMatrix
-   Step 5: Improving the Model
    -   Parameter Tuning
    -   CrossTable
    -   confusionMatrix

Libraries Used
==============

``` r
library(tm) #text mining package from R community, tm_map(), content_transformer()
library(SnowballC) #used for stemming, wordStem(), stemDocument()
library(RColorBrewer) #color palletes
library(wordcloud) #wordcloud generator
library(e1071) #Naive Bayes
library(gmodels) #CrossTable()
library(caret) #ConfusionMatrix()
```

<br />

Objective
=========

As the worldwide use of mobile phones has grown, a new avenue for electronic junk mail has opened for disreputable marketers. These advertisers utilize Short Message Service (SMS) text messages to target potential consumers with unwanted advertising known as SMS spam.

Some examples of spam and ham (non-spam) are as below: ![](smspic.png)

Referring to the table above, there are some clear characteristics that distinguishes spam and ham SMS. For instance, the usage of CAPITAL LETTER, the word "free" and also prices and dates.

Therefore, the objective of this project is to classify a set of SMS messages in text form into either **spam** or **ham (non-spam)** using the **Naive Bayes** machile learning model.

<br />

Step 1: Data Exploration
========================

The data adapted from the SMS Spam Collection at <http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/>. This dataset includes the text of SMS messages along with a label indicating whether the message is unwanted. Junk messages are labeled spam, while legitimate messages are labeled ham.

Data Preview
------------

The dataset Using the head() and str() function, the dataset looks like this:

``` r
head(sms_raw)
```

    ##   type
    ## 1  ham
    ## 2  ham
    ## 3 spam
    ## 4  ham
    ## 5  ham
    ## 6 spam
    ##                                                                                                                                                          text
    ## 1                                             Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
    ## 2                                                                                                                               Ok lar... Joking wif u oni...
    ## 3 Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
    ## 4                                                                                                           U dun say so early hor... U c already then say...
    ## 5                                                                                               Nah I don't think he goes to usf, he lives around here though
    ## 6  FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, \302\2431.50 to rcv

``` r
str(sms_raw)
```

    ## 'data.frame':    5574 obs. of  2 variables:
    ##  $ type: Factor w/ 2 levels "ham","spam": 1 1 2 1 1 2 1 1 2 2 ...
    ##  $ text: chr  "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..." "Ok lar... Joking wif u oni..." "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C"| __truncated__ "U dun say so early hor... U c already then say..." ...

From the preview of the text in the dataset, it is obvious that the source of it is from **Singapore** with slangs such as "lar", "hor", and "jurong point" used.

Features
--------

From the data preview above, the target variable and features can be identified:

### Target Variable

The target variable / class is obviously the type, with levels **spam** and **ham**. The count and proportion are as below:

``` r
table(sms_raw$type)
```

    ## 
    ##  ham spam 
    ## 4827  747

``` r
round(prop.table(table(sms_raw$type)), digits = 2)
```

    ## 
    ##  ham spam 
    ## 0.87 0.13

### Other Features

The features are obviously the SMS **texts** from the dataset. However, text data are challenging to prepare, because it is necessary to transform the words and sentences into a form that a computer can understand.

The data will be transformed into a representation known as **bag-of-words**, which ignores word order and simply provides a variable indicating whether the word appears at all.

Data Visualization
------------------

A great way to visualize the SMS texts would be a **word cloud**. A word cloud depicts the frequency of words appearing. The larget the font, the higher the frequency.

### Spam vs Ham Word Cloud

Now how does the word cloud for Spam vs Ham compare ?

#### Spam

``` r
spam <- subset(sms_raw, type == "spam")
wordcloud(spam$text, max.words = 60, colors = brewer.pal(5, "Dark2"), random.order = FALSE)
```

![](SMS_Spam_files/figure-markdown_github/unnamed-chunk-5-1.png)

#### Ham

``` r
ham <- subset(sms_raw, type == "ham")
wordcloud(ham$text, max.words = 60, colors = brewer.pal(5, "Dark2"), random.order = FALSE)
```

![](SMS_Spam_files/figure-markdown_github/unnamed-chunk-6-1.png)

<br />

Step 2: Data Preprocessing
==========================

The first step in processing text data involves creating a **corpus**, which is a collection of text documents. However, the texts will need to be cleansed and standardized first.

Data Cleasing
-------------

Cleasing involves remove numbers and punctuation; handle uninteresting words such as **and, but, and or**; and how to break apart sentences into individual words. Thankfully, this functionality has been provided by the members of the R community in a text mining package titled **tm**, (text miner).

### Corpus Creation

A **corpus** is created to contain the collection of text documents which contains 5574 SMS Messages

``` r
#Steps to creating a corpus

#Step 1: Prepare a vector source object using VectorSource
#Step 2: Supply the vector source to VCorpus, to import from sources
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

#To view a message, must use double bracket and as.character()
lapply(sms_corpus[1:2], as.character)
```

    ## $`1`
    ## [1] "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."
    ## 
    ## $`2`
    ## [1] "Ok lar... Joking wif u oni..."

### Corpus Cleaning

Before separating the corpus into individual words, the corpus must be cleaned and standardize from uppercase, numbers, punctionation, and clutter characters. The function **tm\_map()** is used.

``` r
# converts to lowercase
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))

#remove numbers as numbers are unique
sms_corpus_clean <- tm_map(sms_corpus_clean, content_transformer(removeNumbers))

#removing stop words, i.e, to, or, but, and. Use stopwords() as argument, parameter that indicates what words we don't want
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())

#remove punctuation, i,e "", .., ', `
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

#apply stemming, removing suffixes f(learns, learning, learned) --> learn
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

#lastly, strip addtional whitespaces
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)
```

### Post Corpus Cleansing

To recap, the following was done to cleanse the corpus:

-   remove numbers
-   convert to lowercase
-   remove punctuation
-   remove stopwords
-   applied stemming
-   remove addtional whitespaces

![](sms2.png)

Data Preparation
----------------

After cleansing, the final step is to split these text messages into individual words through **tokenization**, single element of words. To do this, a **Document Term Matrix (DTM)** is created. The DTM is a that contains clumns of all the words and frequency in each SMS. It is also a **sparse matrix**, where most of it's entries are populated with zeros.

``` r
#convert our corpus to a DTM
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

#dimension of DTM
dim(sms_dtm)
```

    ## [1] 5574 6617

``` r
#alternate way to data cleanse all in 1 go
sms_dtm2 <- DocumentTermMatrix(sms_corpus_clean, control = 
                                 list(tolower = TRUE,
                                      removeNumbers = TRUE,
                                      stopwords = TRUE,
                                      removePunctuation = TRUE,
                                      stemming = TRUE))
```

### Corpus Word Cloud

Wordcloud of the cleansed corpus:

``` r
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE, colors=brewer.pal(8, "Dark2"))
```

![](SMS_Spam_files/figure-markdown_github/unnamed-chunk-10-1.png)

``` r
#min.freq set to 50 implies must appear 50 times before showing in cloud, roughly 1% of total words in DTM
#random order false indicates center in largest
```

### Creating Training and Test Dataset

The dataset will be divided into two portions, **training** and **test** with a percentage of 75 to 25. As the data was already randomly sorted, it can be divided directly.

**Preparing Training and Test Set**

``` r
#Training set
sms_dtm_train <- sms_dtm[1:4180, ]

#Test set
sms_dtm_test <- sms_dtm[4181:5574, ]
```

**Preparing Training and Test Labels**

``` r
#Training Label
sms_train_labels <- sms_raw[1:4180, ]$type

#Test Label
sms_test_labels <- sms_raw[4181:5574, ]$type
```

To ensure the train and test sets are representative, both sets should rougly have the same proportion of spam and ham.

``` r
#Proportion for train labels
prop.table(table(sms_train_labels))
```

    ## sms_train_labels
    ##       ham      spam 
    ## 0.8648325 0.1351675

``` r
#Proportion for test labels
prop.table(table(sms_test_labels))
```

    ## sms_test_labels
    ##       ham      spam 
    ## 0.8694405 0.1305595

### Creating Indicator Features

To transform the **sparse matrix** into something the Naive Bayes classifier can train. If not, all we have is the DTM which are obviously numeric. The frequency of the top appearing words can be obtained using findFreqTerms(). It accepts a DTM and returns a **character vector** the minimum specified frequency.

``` r
# finding words that appear at least 5 times
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)

#preview of most frequent words, 1166 terms with at least 5 occurences
str(sms_freq_words)
```

    ##  chr [1:1166] "<c2><a3>" "<c2><a3>wk" "<c3><9c>" "<c3><bc>" ...

``` r
#filter the DTM sparse matrix to only contain words with at least 5 occurence
#reducing the features in our DTM
sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]
```

Since Naive Bayes trains on categorical data, the numerical data must be converted to categorical data. We need to convert our counts in our 2 sparse matrices into **Yes/No** levels.

``` r
# create a function to do , convert zeros and non-zeros into "Yes" or "No"
convert_counts <- function(x){
  x <- ifelse(x > 0, "Yes", "No")
}

#apply to train and test reduced DTMs, applying to column
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

#check structure of both the DTM matrices
str(sms_train)
```

    ##  chr [1:4180, 1:1166] "No" "No" "No" "No" "No" "Yes" ...
    ##  - attr(*, "dimnames")=List of 2
    ##   ..$ Docs : chr [1:4180] "1" "2" "3" "4" ...
    ##   ..$ Terms: chr [1:1166] "<c2><a3>" "<c2><a3>wk" "<c3><9c>" "<c3><bc>" ...

``` r
str(sms_test)
```

    ##  chr [1:1394, 1:1166] "No" "No" "No" "Yes" "No" "No" ...
    ##  - attr(*, "dimnames")=List of 2
    ##   ..$ Docs : chr [1:1394] "4181" "4182" "4183" "4184" ...
    ##   ..$ Terms: chr [1:1166] "<c2><a3>" "<c2><a3>wk" "<c3><9c>" "<c3><bc>" ...

<br />

Step 3: Model Training
======================

The raw SMS messages has been transformed into a format that can be represented by a statistical model, it is time to apply the Naive Bayes algorithm. The algorithm will use the presence or absence of words to estimate the probability that a given SMS message is spam.

``` r
# applying Naive Bayes to training set
sms_classifier <- naiveBayes(sms_train, sms_train_labels, laplace = 0)

#applying to test set
sms_test_pred <- predict(sms_classifier, sms_test)

#preview of output
head(data.frame("actual" = sms_test_labels, "predicted" = sms_test_pred))
```

    ##   actual predicted
    ## 1    ham       ham
    ## 2    ham       ham
    ## 3    ham       ham
    ## 4   spam      spam
    ## 5    ham       ham
    ## 6    ham       ham

<br />

Step 4: Model Evaluation
========================

To evaluate the accuracy of the Naive Bayes model, CrossTable() and confusionMatrix() is used.

CrossTable
----------

``` r
CrossTable(sms_test_pred, sms_test_labels, prop.chisq = FALSE, dnn = c("predicted", "actual"))
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  1394 
    ## 
    ##  
    ##              | actual 
    ##    predicted |       ham |      spam | Row Total | 
    ## -------------|-----------|-----------|-----------|
    ##          ham |      1205 |        21 |      1226 | 
    ##              |     0.983 |     0.017 |     0.879 | 
    ##              |     0.994 |     0.115 |           | 
    ##              |     0.864 |     0.015 |           | 
    ## -------------|-----------|-----------|-----------|
    ##         spam |         7 |       161 |       168 | 
    ##              |     0.042 |     0.958 |     0.121 | 
    ##              |     0.006 |     0.885 |           | 
    ##              |     0.005 |     0.115 |           | 
    ## -------------|-----------|-----------|-----------|
    ## Column Total |      1212 |       182 |      1394 | 
    ##              |     0.869 |     0.131 |           | 
    ## -------------|-----------|-----------|-----------|
    ## 
    ## 

confusionMatrix
---------------

``` r
confusionMatrix(sms_test_pred, sms_test_labels, dnn = c("predicted", "actual"))
```

    ## Confusion Matrix and Statistics
    ## 
    ##          actual
    ## predicted  ham spam
    ##      ham  1205   21
    ##      spam    7  161
    ##                                           
    ##                Accuracy : 0.9799          
    ##                  95% CI : (0.9711, 0.9866)
    ##     No Information Rate : 0.8694          
    ##     P-Value [Acc > NIR] : < 2e-16         
    ##                                           
    ##                   Kappa : 0.9085          
    ##  Mcnemar's Test P-Value : 0.01402         
    ##                                           
    ##             Sensitivity : 0.9942          
    ##             Specificity : 0.8846          
    ##          Pos Pred Value : 0.9829          
    ##          Neg Pred Value : 0.9583          
    ##              Prevalence : 0.8694          
    ##          Detection Rate : 0.8644          
    ##    Detection Prevalence : 0.8795          
    ##       Balanced Accuracy : 0.9394          
    ##                                           
    ##        'Positive' Class : ham             
    ## 

From the two tables above, the model has a decent **accuracy of nearly 98%**, missing out on 7 messages as ham instead of rightfully classifying it as spam.

<br />

Step 5: Improving the Model
===========================

To improve the model, the laplace parameter in the Naive Bayes function is set to 1. Setting it to 1 will reduce the sparsity effect by ensuring each feature occurs at least once to prevent zero probability from the chain product of Bayes Theorem.

Parameter Tuning
----------------

``` r
sms_classifier <- naiveBayes(sms_train, sms_train_labels, laplace = 1)

sms_test_pred <- predict(sms_classifier, sms_test)
```

CrossTable
----------

``` r
CrossTable(sms_test_pred, sms_test_labels, prop.chisq = FALSE, dnn = c("predicted", "actual"))
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  1394 
    ## 
    ##  
    ##              | actual 
    ##    predicted |       ham |      spam | Row Total | 
    ## -------------|-----------|-----------|-----------|
    ##          ham |      1206 |        24 |      1230 | 
    ##              |     0.980 |     0.020 |     0.882 | 
    ##              |     0.995 |     0.132 |           | 
    ##              |     0.865 |     0.017 |           | 
    ## -------------|-----------|-----------|-----------|
    ##         spam |         6 |       158 |       164 | 
    ##              |     0.037 |     0.963 |     0.118 | 
    ##              |     0.005 |     0.868 |           | 
    ##              |     0.004 |     0.113 |           | 
    ## -------------|-----------|-----------|-----------|
    ## Column Total |      1212 |       182 |      1394 | 
    ##              |     0.869 |     0.131 |           | 
    ## -------------|-----------|-----------|-----------|
    ## 
    ## 

confusionMatrix
---------------

``` r
confusionMatrix(sms_test_pred, sms_test_labels, dnn = c("predicted", "actual"))
```

    ## Confusion Matrix and Statistics
    ## 
    ##          actual
    ## predicted  ham spam
    ##      ham  1206   24
    ##      spam    6  158
    ##                                           
    ##                Accuracy : 0.9785          
    ##                  95% CI : (0.9694, 0.9854)
    ##     No Information Rate : 0.8694          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.901           
    ##  Mcnemar's Test P-Value : 0.001911        
    ##                                           
    ##             Sensitivity : 0.9950          
    ##             Specificity : 0.8681          
    ##          Pos Pred Value : 0.9805          
    ##          Neg Pred Value : 0.9634          
    ##              Prevalence : 0.8694          
    ##          Detection Rate : 0.8651          
    ##    Detection Prevalence : 0.8824          
    ##       Balanced Accuracy : 0.9316          
    ##                                           
    ##        'Positive' Class : ham             
    ## 

Despite slight reduction in accuracy rate, the **false positive** was reduced from 7 to 6. Regradless, the Naive Bayes model was able to **correctly classify 97%** of text messages correctly.
