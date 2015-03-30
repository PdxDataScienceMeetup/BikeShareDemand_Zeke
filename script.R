setwd("C:/Users/ze6ke/Desktop/kaggle/Bikes")

rms<-function(prediction, actual){sqrt(mean((prediction-actual)**2))}

train.column.types <- c("character", #datetime
                       "numeric", #season
                       "factor", #holiday
                       "factor", #workingday
                       "numeric", #weather
                       "numeric", #temp
                       "numeric", #atemp
                       "numeric", #humidity
                       "numeric", #windspeed
                       "numeric", #casual
                       "numeric", #registered
                       "numeric") #count


#tell r how to store the imported file.
train<-read.csv("train.csv", colClasses=train.column.types)
#test<-read.csv("test.csv")


train$datetime<-strptime(train$datetime, format="%Y-%m-%d %H:%M:%S")
train$hour<-train$datetime$hour
train$date<-as.POSIXct(format(train$datetime, "%Y-%m-%d"), format="%Y-%m-%d")
train$wday<-train$datetime$wday
#to simplify calculations, for any holidays, I ignore the real day of the week and consider them to be the 8th day
train$wday[train$holiday==1]<-7
train$wday<-factor(train$wday, labels=c("Sun","Mon","Tue","Wed","Thu","Fri","Sat","Hol"))
levels(train$season)<-c("Spring","Summer", "Fall", "Winter")
train$weather<-factor(train$weather, labels=c("Nice","Misty", "Rainy", "Ugly"))
train$lcount<-log(train$count+1)
train$lregistered<-log(train$registered+1)
train$lcasual<-log(train$casual+1)


#train$gcount is the y value, train$groupingField is the grouping field
calculate.and.plot.based.on.groupingField<-function(numpoints=10000)
{
  train.hourly.average<-aggregate(train$gcount, by=list(train$groupingField), FUN=mean)
  plot(train$datetime[1:numpoints], 
       train$gcount[1:numpoints]-
         train.hourly.average[match(train$groupingField[1:numpoints], train.hourly.average[,1]),2], type='l')
  
}

#all days are the same, straight count
train$groupingField<-train$hour
train$gcount<-train$count
calculate.and.plot.based.on.groupingField()

#all days are the same, log of count
train$groupingField<-train$hour
train$gcount<-train$lcount
calculate.and.plot.based.on.groupingField()

#track each day of week separately, log of count
train$groupingField<-as.numeric(train$wday)*24+train$hour
train$gcount<-train$lcount
calculate.and.plot.based.on.groupingField()

#the baseline is my guess of performance based exclusively on time of day and day of week.
#for each point in time, this functions calculates the moving average of that hour of the week
#for the data before the point.  It stores that result with the point in time.
set.baseline<-function(attenuation.factor)
{
  running.average<-rep(NA_real_, 8*24+24)
  baseline<-rep(NA_real_, nrow(train))
  for(i in 1:nrow(train))
  {
    row<-train[i,]
    baseline[i]<-running.average[as.numeric(row$wday)*24+row$hour]
    train$baseline[i]<-running.average[as.numeric(row$wday)*24+row$hour]
    if(is.na(running.average[as.numeric(row$wday)*24+row$hour]))
    {
      running.average[as.numeric(row$wday)*24+row$hour]<-row$gcount 
    }
    else
    {
      running.average[as.numeric(row$wday)*24+row$hour]<-
        (running.average[as.numeric(row$wday)*24+row$hour]*attenuation.factor + 
          row$gcount)/(1+attenuation.factor)
    }
  }  
  return (baseline)
}

#train$baseline<-set.baseline(1)

#this function attempts to improve on the baseline estimate by using multiple linear models.
#It divides the data into number.of.models+1 contiguous blocks, build models on all but the last 
#block and then uses the models from each block to make predictions for the next block.
#e.g., The model from block 12 is used to predict block 13.
#it skips the first block because there is no baseline.
build.linear.predictions<-function(number.of.models)
{
  prediction<-rep(NA_real_, nrow(train))
  number.of.models<-as.numeric(number.of.models)
  number.of.records.per.group<-nrow(train)/(number.of.models+1)
  for(i in 1:(number.of.models-1))#skip first set because we didnt have a baseline
  {
    train.start<-0+(i*number.of.records.per.group)
    cutoff<-train.start + number.of.records.per.group
    test.end<-cutoff + number.of.records.per.group
    
    model<-lm(gcount-baseline~as.numeric(date)+season+weather+temp+atemp+humidity+windspeed, 
              data=train[train.start:cutoff,])
    prediction[(cutoff+1):test.end]<-predict(model, newdata=train[(cutoff+1):test.end,])
  }
  prediction<-prediction+train$baseline
  return(prediction)
}



#this block prepares the features for random forest prediction.  Because RF is good at 
#feature selection and the features are fairly easy to generate, we provide a lot of them.
#we don't keep the straight average because the contest rules say that you can only use historical
#data.  FWIW, the performance of RF is linear on number of features (and nlog n on the number of
#observations)
train$gcount<-train$lcount
train.hourly.averages<-aggregate(train$gcount, by=list(as.numeric(train$wday)*24+train$hour), FUN=mean)
#train$baseline0<-train.hourly.averages[match(as.numeric(train$wday)*24+train$hour, train.hourly.averages[,1]),2]
train$baseline1<-set.baseline(1)
train$baseline2<-set.baseline(2)
train$baseline5<-set.baseline(5)
train$baseline10<-set.baseline(10)
train$baseline25<-set.baseline(25)
train$baseline50<-set.baseline(50)
train$baseline100<-set.baseline(100)
train$baseline250<-set.baseline(250)

library(randomForest)


#this function attempts to improve on the baseline estimate by using multiple random forest models.
#It divides the data into number.of.models+1 contiguous blocks, build models on all but the last 
#block and then uses the models from each block to make predictions for the next block.
#e.g., The model from block 12 is used to predict block 13.
#it skips the first few blocks because of issues with NAs in the baselines.
#if train.on.all.data is true, the function doesn't 12 to predict 13, but instead uses 1-12
#to predict 13.
build.rf.predictions<-function(number.of.models, train.on.all.data=F)
{
  prediction<-rep(NA_real_, nrow(train))
  number.of.models<-as.numeric(number.of.models)
  number.of.records.per.group<-nrow(train)/(number.of.models+1)
  for(i in 3:(number.of.models-1))#skip first two sets because we didn't have a baseline and NAs aren't ok.
  {
    train.start<-0+(i*number.of.records.per.group)
    cutoff<-train.start + number.of.records.per.group
    test.end<-cutoff + number.of.records.per.group
    
    first.training.record<-train.start
    if(train.on.all.data)
    {
      first.training.record<-500
    }
    #print(first.training.record)
    train.data<-train[first.training.record:cutoff,
                      !(names(train) %in% c("prediction", "casual", "registered", "count", "lcasual", "lregistered", "lcount", "gcount"))]
    #print(sapply(train.data, function(x) sum(is.na(x))))
    test.data<-train[(cutoff+1):test.end,
                     !(names(train) %in% c("prediction", "casual", "registered", "count", "lcasual", "lregistered", "lcount", "gcount"))]
    
    train.labels<-train[first.training.record:cutoff,
                        (names(train) %in% c("gcount"))]
    
    model<-randomForest(x=train.data, y=train.labels, ntree=1000, keep.forest=TRUE)
    prediction[(cutoff+1):test.end]<-predict(model, newdata=test.data)
    #this statement isn't important, but it's not gratifying to wait days without
    #any feedback.
    print rms(train$gcount[!is.na(prediction)], na.omit(prediction))
  }
  return(prediction)
}

#prediction<-build.rf.predictions(30, T)

#try it out.  The closest I could get to cross validation was varying the number of periods we
#break the data into.  So, we try 30 different ones.  The two blocks vary based on the
#value of train.on.all.data
errors<-rep(NA_real_, 31)
for(i in 1:31)
{
  prediction<-build.rf.predictions(i+30)
  train$prediction<-prediction
  errors[i]<-rms(train$gcount[!is.na(train$prediction)], na.omit(train$prediction))
}

summary(errors)
#.38 for 49 models and baseline 0-10
#.373-.411 .395 for 31-61 models and baseline 0-10
#.391-.425 .410 for 31-61 models based on small windows and baseline 1-250

errors<-rep(NA_real_, 31)
for(i in 1:31)
{
  prediction<-build.rf.predictions(i+30, T)
  train$prediction<-prediction
  errors[i]<-rms(train$gcount[!is.na(train$prediction)], na.omit(train$prediction))
}

summary(errors)

#.349-.372 .360 for 31-61 models based on all available data and baseline 1-250
#54 hrs



#Everything after this was an expirement




#show the residual after removing baseline predictions with various attenuation rates.
# sds<-rep(NA_real_,5)
# in.sample.error.baseline<-rep(NA_real_,5)
# in.sample.error.lm<-rep(NA_real_,5)
# out.of.sample.error.lm<-rep(NA_real_,5)
# for(i in 1:10)
# {
#   print(i)
#   attenuation.rate<-.2 * 2**i
#   returnval<-set.baseline(attenuation.rate)
#   train$baseline<-returnval
#   #train$prediction<-returnval[[2]]
#   #plot(train$datetime[1:10000], train$gcount[1:10000]-train$baseline[1:10000], type='l')
#   #summary(train$gcount[1:10000]-train$baseline[1:10000])
#   sds[i]<-sd((train$gcount[1:10000]-train$baseline[1:10000]), na.rm=TRUE)
#   model<-lm(gcount-baseline~as.numeric(datetime)+weather+humidity+holiday+atemp+windspeed, data=train)
#   plot(train$datetime[!is.na(train$baseline)], predict(model, data=train)-na.omit(train$gcount-train$baseline), type="l")
#   in.sample.error.lm[i]<-rms(predict(model, data=train)+na.omit(train$baseline), train$gcount[!is.na(train$baseline)])
#   in.sample.error.baseline[i]<-rms(na.omit(train$baseline), train$gcount[!is.na(train$baseline)])
#   predictions<-build.predictions(100)
#   out.of.sample.error.lm[i]<-rms(na.omit(predictions), train$gcount[!is.na(predictions)] )
# }
# sds
# in.sample.error.baseline
# in.sample.error.lm
# out.of.sample.error.lm



#plot days with differing colors based on other variables
# train.daily.total<-aggregate(train$count, by=list(train$date), FUN=sum)
# plot(train.daily.total[,1], runmed(train.daily.total[,2], k=7), type='l')
# aggregate(train$count, by=list(train$date), FUN=sum)[]

# green<-0
#   #wdayi/length(levels(train$wday))
# 
# for(wdayi in 1:length(levels(train$wday)))
# {
#   plot(NULL, NULL, type="n", xlim=c(0,23), ylim=c(0,10))
#   wday<-levels(train$wday)[wdayi]
#   for(seasoni in 1:length(levels(train$season)))
#   {
#     season<-levels(train$season)[seasoni]
#     red<-seasoni*1.0/length(levels(train$season))
#    if((season %in% c("Spring")))
#     {
#      for(weatheri in 1:length(levels(train$weather)))
#       {
#         weather<-levels(train$weather)[weatheri]
#         blue<-weatheri*1.0/length(levels(train$weather))
#         
#         print(c(red, green, blue))
#         for(date in unique(train$date[train$wday==wday&train$season==season&train$weather==weather]))
#         {
#           
#           lines(train$hour[train$date==date], 
#                 train$lcount[train$date==date], 
#                 col=rgb(red, green, blue, alpha=.3))
#         }
#       }
#      }
#   }
# }

#summary(train$datetime)
#summary(train$datetime<"2012-01-01")

#do a simple rf model choping the data up based on year or based on day of the month.
#train.data<-train[train$datetime<"2012-01-01",][c(-10:-12, -16:-18)]
#train.labels<-train[train$datetime<"2012-01-01",][[16]] #pull vector out of list

#test.data<-train[train$datetime>="2012-01-01",][c(-10:-12, -16:-18)]
#test.labels<-train[train$datetime>="2012-01-01",][16]


# train.data<-train[train$datetime$mday<11&train$datetime$year==111,][c(-10:-12, -16:-18)]
# train.labels<-train[train$datetime$mday<11&train$datetime$year==111,][[16]] #pull vector out of list
# 
# test.data<-train[train$datetime$mday>=11&train$datetime$year==111,][c(-10:-12, -16:-18)]
# test.labels<-train[train$datetime$mday>=11&train$datetime$year==111,][16]

#library(randomForest)
#rf <- randomForest(x=train.data, y=train.labels, ntree=1000, keep.forest=TRUE)
#predictions <-predict(rf, test.data)
#rms(predictions, test.labels)
#.576 for by year
# .408 for by month
# .450 for by month and only 2011