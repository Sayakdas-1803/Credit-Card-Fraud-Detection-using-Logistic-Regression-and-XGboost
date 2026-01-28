# CREDIT CARD FRAUD DETECTION PROJECT (R)
# Models:
# 1. Logistic Regression (class-weighted)
# 3. XGBoost (scale_pos_weight)



#Load required libraries...................
library(dplyr)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)
library(xgboost)

#Load dataset.........................
data <-read.csv("C:\\Users\\sayak\\Downloads\\creditcard.csv\\creditcard.csv")
str(data)


#Exploratory Data Analysis............................

# Check class imbalance
table(data$Class)
prop.table(table(data$Class))

sum(is.na(data))  #checking if there is any NA value

#Separate fraud and non fraud into new files............
data.true<-data[data$Class==0,]
data.false<-data[data$Class==1,]

#data visualization........................
ggplot()+geom_density(data=data.true,aes(x=Time),color="blue",fill="blue",
                      alpha=0.12)+
         geom_density(data=data.false,aes(x=Time),color="red",fill="red",
                      alpha=0.12)
#This graph shows that there is more number of frauds in the start 
#of the month

ggplot(data, aes(x = Class)) +
  geom_bar(fill = c("blue", "red")) +
  labs(title = "Class Distribution (0: No Fraud, 1: Fraud)",
       x = "Class", y = "Count") +
  theme_minimal()

# Check the distribution of transaction amounts......................
ggplot()+geom_density(data=data.true,aes(x=Amount),color="blue",fill="blue",
                      alpha=0.2)+
         geom_density(data=data.false,aes(x=Amount),color="red",fill="red",
                      alpha=0.2)
#This shows that the amount when higher probability of it to be 
#fraud is also higher.

#Data preprocessing...........................

# Convert target to factor
data$Class <- factor(data$Class, levels = c(0,1))

# Scale Time and Amount
data$Time <- scale(data$Time)
data$Amount <- scale(data$Amount)

#Train-Test Split
set.seed(123)
train_index <- createDataPartition(data$Class, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

#LOGISTIC REGRESSION
log_model <- glm(Class ~ .,
                 data = train_data,
                 family = binomial)


summary(log_model)

# Predict probabilities
log_prob <- predict(log_model, test_data, type = "response")
log_pred <- factor(ifelse(log_prob > 0.3, 1, 0), levels = c(0,1))

confusionMatrix(log_pred, test_data$Class)

roc_log <- roc(as.numeric(test_data$Class), log_prob)
auc(roc_log)


#XGBOOST.........................
# Prepare matrix data
x_train <- as.matrix(select(train_data, -Class))
y_train <- as.numeric(train_data$Class) - 1

x_test  <- as.matrix(select(test_data, -Class))
y_test  <- as.numeric(test_data$Class) - 1

# Calculate imbalance ratio
neg_pos_ratio <- sum(y_train == 0) / sum(y_train == 1)

# DMatrix objects
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test,  label = y_test)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 5,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = neg_pos_ratio
)

set.seed(123)
xgb_model <- xgb.train(params = params,
                       data = dtrain,
                       nrounds = 200,
                       verbose = 0)

xgb_prob <- predict(xgb_model, dtest)
xgb_pred <- factor(ifelse(xgb_prob > 0.3, 1, 0), levels = c(0,1))

confusionMatrix(xgb_pred, test_data$Class)

roc_xgb <- roc(as.numeric(test_data$Class), xgb_prob)
auc(roc_xgb)


#ROC CURVES (Visualization)......................
# ROC Curve: Logistic Regression...........................
plot(roc_log, col = "blue", lwd = 2,
     main = "ROC Curve - Logistic Regression")

# ROC Curve: XGBoost..........................
plot(roc_xgb, col = "red", lwd = 2,
     main = "ROC Curve - XGBoost")

#Combined ROC Curve Comparison...........................
plot(roc_log, col = "blue", lwd = 2,
     main = "ROC Curve Comparison")
lines(roc_xgb, col = "red", lwd = 2)

legend("bottomright",
       legend = c("Logistic Regression", "XGBoost"),
       col = c("blue", "red"),
       lwd = 2)

#Model Comparison (AUC Values)
auc_log  <- auc(roc_log)
auc_xgb  <- auc(roc_xgb)

auc_log
auc_xgb


#Conclusion
#Logistic Regression: Interpretable, improved recall via class weights
#XGBoost: Best performance with imbalance-aware training
