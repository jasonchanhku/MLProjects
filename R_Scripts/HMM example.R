#HMM implementation example

#libraries loaded
library(depmixS4)
library(quantmod)
library(plotly)
#load data
data <- read.csv("My Received Files/EURUSD1d.csv", stringsAsFactors = FALSE)
View(data)

#preprocessing
#convert to time series
Date <- as.character(data[ , 1])
Date_TS <- as.POSIXct(Date, format = "%Y.%m.%d %H:%M:%S")
TS <- data.frame(data[ , 2:5], row.names = Date_TS)
TS_xts <- as.xts(TS)
#quick look
plot_ly(TS, x = ~row.names(TS), y = ~TS$Close,  type = "scatter", mode = "lines")

#calculating ATR
ATR <- ATR(TS_xts[ ,2:4], n = 14)
ATR_data <- ATR[ , 2]

#calculating log returns
logret <- log(TS$Close) - log(TS$Open)

#creating data frame
model <- data.frame(logret, ATR_data)
model <- model[-c(1:14), ]
colnames(model) <- c("LogReturns", "ATR")
View(model)

#Implementing a multiple HMM, each HMM a regime
#Setting LogRet and ATR as the response variables
#Set 3 different regimes, distribution equal to gaussian
hmm <- depmix(list(LogReturns~1, ATR~1), data = model, nstates = 3, family = list(gaussian(), gaussian()))

#optimize parameters
hmm_fit <- fit(hmm)
hmm_fit

#printing out transition matrix
summary(hmm_fit)

#posterior probability, obs 1, given states 1,2,3
hmm_pos <- posterior(hmm_fit)
head(hmm_pos)

#plotting
DFIndicators <- data.frame(Date_TS, logret, ATR_data); 
DFIndicatorsClean <- DFIndicators[-c(1:14), ]

Plot1Data<-data.frame(DFIndicatorsClean, hmm_pos$state)

#log returns
plot_ly(Plot1Data, x = ~Date_TS, y = ~logret, type = "scatter", mode = "lines")

#ATR
plot_ly(Plot1Data, x = ~Date_TS, y = ~atr, type = "scatter", mode = "lines")

#States
plot_ly(Plot1Data, x = ~Date_TS, y = ~hmm_pos.state, type = "scatter", mode = "lines")

#Regime probabilities
RegimePlotData<-data.frame(Plot1Data$Date_TS,hmm_pos)
View(RegimePlotData)

#All regimes
plot_ly(RegimePlotData, x = ~RegimePlotData[ , 1], y = ~S1, type = "scatter", mode = "lines", name = "Low") %>% add_trace(y = ~S2, name = "Medium", mode = "lines") %>% add_trace(y = ~S3, name = "High", mode = "lines")
