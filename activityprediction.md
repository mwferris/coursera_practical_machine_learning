---
title: "ActivityPrediction"
author: "Mark Ferris"
date: "April 12, 2018"
output: html_document
---



## Introduction
The goal of this assignment is to use machine learning to predict if a subject performed a weight lifting exercise correctly. The data comes from various sensors placed on the subject and dumbell. The subject either performs the lift correctly (classe A) or incorrectly (error classes B, C, D, and E).

All data comes from the Human Activity Recognition (HAR) group of the Rio de Janeiro PUC university - http://groupware.les.inf.puc-rio.br/har

## Getting Data

#load required packages

```r
library(caret)
```
# Download data
First the data is downloaded and cleaned up. It was observed that columns containing NAs had too many NA values in them to impute, so they were discared from the training and testing sets.

```r
trainingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(trainingUrl, destfile = "./data/training.csv")
download.file(testingUrl, destfile = "./data/testing.csv")

# Read data
training <- read.csv("./data/training.csv", na.strings = c("NA", "#DIV/0!"))
testing <- read.csv("./data/testing.csv", na.strings = c("NA", "#DIV/0!"))

#first remove first seven columns as they are indicators not predictors

training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]

#store training columns with NA values
naCols <- which(colSums(is.na(training)) > 0, arr.ind=TRUE, useNames=FALSE)

#remove from training
training <- training[,-c(naCols)]

#remove from testing
testing <- testing[,-c(naCols)]
```
Then the training set was checked for near zero variables, which there were none of

```r
# check for near zero variables (all contribute)
nearZeroVar(training, saveMetrics = TRUE)
```

```
##                      freqRatio percentUnique zeroVar   nzv
## roll_belt             1.101904     6.7781062   FALSE FALSE
## pitch_belt            1.036082     9.3772296   FALSE FALSE
## yaw_belt              1.058480     9.9734991   FALSE FALSE
## total_accel_belt      1.063160     0.1477933   FALSE FALSE
## gyros_belt_x          1.058651     0.7134849   FALSE FALSE
## gyros_belt_y          1.144000     0.3516461   FALSE FALSE
## gyros_belt_z          1.066214     0.8612782   FALSE FALSE
## accel_belt_x          1.055412     0.8357966   FALSE FALSE
## accel_belt_y          1.113725     0.7287738   FALSE FALSE
## accel_belt_z          1.078767     1.5237998   FALSE FALSE
## magnet_belt_x         1.090141     1.6664968   FALSE FALSE
## magnet_belt_y         1.099688     1.5187035   FALSE FALSE
## magnet_belt_z         1.006369     2.3290184   FALSE FALSE
## roll_arm             52.338462    13.5256345   FALSE FALSE
## pitch_arm            87.256410    15.7323412   FALSE FALSE
## yaw_arm              33.029126    14.6570176   FALSE FALSE
## total_accel_arm       1.024526     0.3363572   FALSE FALSE
## gyros_arm_x           1.015504     3.2769341   FALSE FALSE
## gyros_arm_y           1.454369     1.9162165   FALSE FALSE
## gyros_arm_z           1.110687     1.2638875   FALSE FALSE
## accel_arm_x           1.017341     3.9598410   FALSE FALSE
## accel_arm_y           1.140187     2.7367241   FALSE FALSE
## accel_arm_z           1.128000     4.0362858   FALSE FALSE
## magnet_arm_x          1.000000     6.8239731   FALSE FALSE
## magnet_arm_y          1.056818     4.4439914   FALSE FALSE
## magnet_arm_z          1.036364     6.4468454   FALSE FALSE
## roll_dumbbell         1.022388    84.2065029   FALSE FALSE
## pitch_dumbbell        2.277372    81.7449801   FALSE FALSE
## yaw_dumbbell          1.132231    83.4828254   FALSE FALSE
## total_accel_dumbbell  1.072634     0.2191418   FALSE FALSE
## gyros_dumbbell_x      1.003268     1.2282132   FALSE FALSE
## gyros_dumbbell_y      1.264957     1.4167771   FALSE FALSE
## gyros_dumbbell_z      1.060100     1.0498420   FALSE FALSE
## accel_dumbbell_x      1.018018     2.1659362   FALSE FALSE
## accel_dumbbell_y      1.053061     2.3748853   FALSE FALSE
## accel_dumbbell_z      1.133333     2.0894914   FALSE FALSE
## magnet_dumbbell_x     1.098266     5.7486495   FALSE FALSE
## magnet_dumbbell_y     1.197740     4.3012945   FALSE FALSE
## magnet_dumbbell_z     1.020833     3.4451126   FALSE FALSE
## roll_forearm         11.589286    11.0895933   FALSE FALSE
## pitch_forearm        65.983051    14.8557741   FALSE FALSE
## yaw_forearm          15.322835    10.1467740   FALSE FALSE
## total_accel_forearm   1.128928     0.3567424   FALSE FALSE
## gyros_forearm_x       1.059273     1.5187035   FALSE FALSE
## gyros_forearm_y       1.036554     3.7763735   FALSE FALSE
## gyros_forearm_z       1.122917     1.5645704   FALSE FALSE
## accel_forearm_x       1.126437     4.0464784   FALSE FALSE
## accel_forearm_y       1.059406     5.1116094   FALSE FALSE
## accel_forearm_z       1.006250     2.9558659   FALSE FALSE
## magnet_forearm_x      1.012346     7.7667924   FALSE FALSE
## magnet_forearm_y      1.246914     9.5403119   FALSE FALSE
## magnet_forearm_z      1.000000     8.5771073   FALSE FALSE
## classe                1.469581     0.0254816   FALSE FALSE
```

## Cross Validation
The training set was then split into new training and testing data to determine which model was appropriate to use.


```r
## Split training into new train and test
inTrain <- createDataPartition(y=training$classe, p=0.75, list=FALSE)
subTraining <- training[inTrain,]
subTesting <- training[-inTrain,]
```
Boosting took a significant amount of time so it was abandoned. The random forest method was selected to attempt next. To improve model performance (decrease runtime to train) preprocessing with PCA was performed.


The model was used to predict on the testing subgroup of the training data set

```r
pred <- predict(modFit, subTesting);
```
Which was then evaluated using Caret's confusionMatrix function

```r
confusionMatrix(pred, subTesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    2    1    0    0
##          B    0  944    3    0    0
##          C    0    2  851    5    2
##          D    0    0    0  799    0
##          E    0    1    0    0  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9967          
##                  95% CI : (0.9947, 0.9981)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9959          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9947   0.9953   0.9938   0.9978
## Specificity            0.9991   0.9992   0.9978   1.0000   0.9998
## Pos Pred Value         0.9979   0.9968   0.9895   1.0000   0.9989
## Neg Pred Value         1.0000   0.9987   0.9990   0.9988   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1925   0.1735   0.1629   0.1833
## Detection Prevalence   0.2851   0.1931   0.1754   0.1629   0.1835
## Balanced Accuracy      0.9996   0.9970   0.9965   0.9969   0.9988
```
This was highly accurate, with a 0.9724 kappa value.

## Predicting on Real Test

```r
testPred <- predict(modFit, testing)
print(testPred)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

These values were submitted to the validation quiz and were 100% correct.
