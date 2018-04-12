#set working directory
setwd("C:/Users/mferris/OneDrive/Documents/Coursera/Data Science/Practical Machine Learning/Prediction Assignment Writeup")

#load required packages
library(caret)

# Download data
trainingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(trainingUrl, destfile = "./data/training.csv")
download.file(testingUrl, destfile = "./data/testing.csv")

# Read data
training <- read.csv("./data/training.csv", na.strings = c("NA", "#DIV/0!"))
testing <- read.csv("./data/testing.csv", na.strings = c("NA", "#DIV/0!"))

#preprocess data

#first remove first seven columns as they are indicators not predictors

training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]

#store training columns with NA values
naCols <- which(colSums(is.na(training)) > 0, arr.ind=TRUE, useNames=FALSE)

#remove from training
training <- training[,-c(naCols)]

#remove from testing
testing <- testing[,-c(naCols)]

