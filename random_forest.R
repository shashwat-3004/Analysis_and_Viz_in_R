library(tidyverse)
library(caret)
library(caTools)
library(randomForest)
library(e1071)
library(xgboost)

data<-read_csv("oasis_longitudinal.csv")

data<-data[,c(-1,-2,-4,-7,-12,-15)]
str(data)

data<-data%>%filter(Group!="Converted")
summary(data)

data<-data%>%mutate_all(~ifelse(is.na(.), median(., na.rm = TRUE), .))

data$Group<-as.factor(data$Group)
data$`M/F`<-as.factor(data$`M/F`)

colnames(data)<-c("Group","MR_delay","Gender","Age","EDUC","SES","MMSE","eTIV","nWBV")
set.seed(123)

split_dat<-sample.split(data$Group,SplitRatio = 0.7)

train_data<-data[split_dat,]
test_data<-data[!split_dat,]


trControl <- trainControl(method = "cv",
                          number = 5,
                          search = "random")

mtry<-sqrt(ncol(train_data))

rf_random <- train(Group ~ .,
                   data = train_data,
                   method = 'rf',
                   metric = 'Accuracy',
                   tuneLength  = 15, 
                   trControl = trControl)
print(rf_random)

plot(rf_random)


modellist <- list()

#train with different ntree parameters
for (ntree in c(25,50,100,150,200,250,300)){
  fit <- train(Group~.,
               data = train_data,
               method = 'rf',
               metric = 'Accuracy',
               tuneGrid = NULL,
               trControl = trControl,
               ntree = ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}

#Compare results
results <- resamples(modellist)
summary(results)

dotplot(results)


classifier_RF<-randomForest(Group ~ ., 
                            data = train_data, 
                            importance = TRUE,
                            proximity = TRUE,
                            ntree=150,
                            )
classifier_RF

train_pred<-predict(classifier_RF,train_data)


test_pred<-predict(classifier_RF,test_data)

train_rf_CM<-confusionMatrix(train_pred,train_data$Group)
test_rf_CM<-confusionMatrix(test_pred,test_data$Group)

plot(classifier_RF)

varImpPlot(classifier_RF)


-------------------------------------------------
# Logistic Regression

logistic_model <- glm(Group~., 
                      data = train_data, 
                      family = "binomial")  
summary(logistic_model)

train_log_pred<-ifelse(predict(logistic_model,train_data)>0.5,1,0)

test_log_pred<-ifelse(predict(logistic_model,test_data)>0.5,1,0)

train_log_CM<-confusionMatrix(as.factor(train_log_pred),as.factor((as.numeric(train_data$Group)-1)))
test_log_CM<-confusionMatrix(as.factor(test_log_pred),as.factor((as.numeric(test_data$Group)-1)))


------------------------------------------
#XGboost  

dtrain = xgb.DMatrix(as.matrix(sapply(train_data, as.numeric)), label=(as.numeric(train_data$Group)-1))

dest = xgb.DMatrix(as.matrix(sapply(test_data, as.numeric)), label=(as.numeric(test_data$Group)-1))

xg_classifier<-xgboost(data = dtrain, nthread = 2, nrounds = 20, objective = "binary:logistic", verbose = 2)

xg_predict_test<-ifelse(predict(xg_classifier,dest)>0.5,1,0)

XG_boost_test_CM<-confusionMatrix(as.factor(xg_predict_test),as.factor((as.numeric(test_data$Group)-1)))
