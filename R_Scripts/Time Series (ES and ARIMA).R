#Libraries Used
library(forecast)

#Time Series Analysis
births <- scan("http://robjhyndman.com/tsdldata/data/nybirths.dat")

births <- ts(births, frequency = 12, start = c(1946, 1))
births

gift <- scan("http://robjhyndman.com/tsdldata/data/fancy.dat")
gift<- ts(gift, frequency=12, start=c(1987,1))

plot.ts(gift)
logGift <- log(gift)
plot.ts(logGift)

kings<-scan('http://robjhyndman.com/tsdldata/misc/kings.dat', skip=3)
kings <- ts(kings)
plot.ts(kings)

kingsSMA3 <- SMA(kings, n=8)
plot.ts(kingsSMA3)

birthsComp <- decompose(births)
plot(birthsComp)
birthsComp$seasonal
birthsComp$trend
birthsComp$random

#Forecast using simple exponential smoothing
rain <- scan("http://robjhyndman.com/tsdldata/hurst/precip1.dat",skip=1)
rainseries <- ts(rain, start = c(1813))
plot.ts(rainseries)

# parameter is alpha, 0 < a < 1, 0 means little weight is placed on the most recent 
# observations when making forecasts of future values.
# beta false means exponential smoothing
# gamma false means non-seasonal model fitted
rainseriesforecasts <- HoltWinters(rainseries, beta = FALSE, gamma = FALSE)
rainseriesforecasts2 <- forecast.HoltWinters(rainseriesforecasts, h = 8)
plot.forecast(rainseriesforecasts2)

#model improvement
# If the predictive model cannot be improved upon, 
# there should be no correlations between forecast errors for successive predictions
# remove NA in first value

# 2 things to check to see if model is sufficient:
# 1) acf, not surpassing the significance bound
# 2) residuals are normally distributed with constant variance and mean 0
#

acf(rainseriesforecasts2$residuals[-1])

#Holt's Exponential Smoothing

skirts <- scan("http://robjhyndman.com/tsdldata/roberts/skirts.dat",skip=5)
skirtsseries <- ts(skirts, start = c(1866))
plot.ts(skirtsseries)

#Fitting Holt's Exponential Smoothing
skirtsseriesforecasts <- HoltWinters(skirtsseries, gamma = FALSE)
skirtsseriesforecasts$SSE

plot(skirtsseriesforecasts)

skirtsseriesforecasts2 <- forecast.HoltWinters(skirtsseriesforecasts, h = 19)
plot(skirtsseriesforecasts2)

#make sure no corr and make sure residuals are normally dist with 0 mean constant var
acf(na.trim.ts(skirtsseriesforecasts2$residuals))

plot(skirtsseriesforecasts2$residuals)

#Holt Winter's Exponential Smoothing

logsouvenirts <- logGift
souvenirtsforecast <- HoltWinters(logsouvenirts)
souvenirtsforecast

souvenirtsforecast$SSE

souvenirtsforecast2 <- forecast.HoltWinters(souvenirtsforecast, h = 48)
plot.forecast(souvenirtsforecast2)

acf(na.trim.ts(souvenirtsforecast2$residuals))

#ARIMA
#Step 1: differencing until stationary, take out trend component
skirtsseriesdiff1 <- diff(skirtsseries, differences = 1)
plot.ts(skirtsseriesdiff1)

skirtsseriesdiff2 <- diff(skirtsseries, differences = 2)
plot.ts(skirtsseriesdiff2)

#kings ARIMA example
kingsdiff1 <- diff(kings, differences = 1)
plot.ts(kingsdiff1)

#Step 2: Check correlogram (ACF and PACF) of irregular term
#ACF
acf(kingsdiff1, lag.max = 20)
acf(kingsdiff1, lag.max = 20, plot = FALSE)

#PACF
pacf(kingsdiff1, lag.max = 20)
pacf(kingsdiff1, lag.max = 20, plot = FALSE)

#Forecasting ARIMA
kingsarima <- arima(kings, order = c(0,1,1))
kingsarimaforecast <- forecast.Arima(kingsarima, h = 5)
plot.forecast(kingsarimaforecast)
kingsarimaforecast

#Check for ACF and normal dist
acf(kingsarimaforecast$residuals)
plot.ts(kingsarimaforecast$residuals)
hist(kingsarimaforecast$residuals)

# Volcano ARIMA example
volcanodust <- scan("http://robjhyndman.com/tsdldata/annual/dvi.dat", skip=1)
volcanodustseries <- ts(volcanodust, start = c(1500))
plot.ts(volcanodustseries)

#check if difference needed 
auto.arima(volcanodustseries, ic = "bic")

#check ACF and PACF
acf(volcanodustseries, lag.max = 20)
pacf(volcanodustseries, lag.max = 20, plot = FALSE)

#forecast ARIMA(2,0,0)
volarima <- arima(volcanodustseries, order = c(2,0,0))
volforecast <- forecast.Arima(volarima, h = 5)
plot.forecast(volforecast)
volforecast

#check acf of forecast residuals and normal 
acf(volforecast$residuals)
plot(volforecast$residuals)
