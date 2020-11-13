# Road Map
# .	Library for Preprocessing and Cleaning
# .	Load all Classification Packages and Accuracy Packages
# .	Load Data Set
# .	Analyse the Data
# .	LabelEncoder
# .	Split the Data Train and Validation
# .	Train Model and Check Validation Data Accuracy

#Objective:The Importance of Predicting Customer Churn
#The ability to predict that a particular customer is at a high risk of churning, while there is still time
# to do something about it, represents a huge additional potential revenue source for every online business. 
# Besides the direct loss of revenue that results from a customer abandoning the business, the costs of initially 
# acquiring that customer may not have already been covered by the customers spending to date. 
# (In other words, acquiring that customer may have actually been a losing investment.) Furthermore, 
# it is always more difficult and expensive to acquire a new customer than it is to retain a current paying customer.
#########################################################################
#Libraries used for Classification models, Preprocessing and Cleaning
install.packages('corrplot')
library(pROC)
library(MASS)
library(randomForest)
library(e1071)
library(ISLR)
library(caret)
library(leaps)
library(class)
library(corrplot)
#caret package provides us direct access to various functions for training our model with various machine learning algorithms 
#like Knn, SVM, random Forest, logistic regression, etc.
##########################################################################
#Load the data set
churn.dat<-read.csv("C://Users//Roofiya Koya//Documents//Sem 2//CS6405//churn//churn-train.csv",header=TRUE)
head(churn.dat)
attach(churn.dat)
table(class)
#Check for missing values
col=ncol(churn.dat)
row=nrow(churn.dat)
missing= sum(is.na(churn.dat))

##########################################################################
#Preliminary Analyses on the Churn Dataset
churn.dat$state=as.factor(state)
churn.dat$area_code=as.factor(area_code)
churn.dat$international_plan=as.factor(international_plan)
churn.dat$voice_mail_plan=as.factor(voice_mail_plan)
churn.dat$class=as.factor(class)

levels(churn.dat$class)<-make.names(c('NoChurn','Churn'))
str(churn.dat)
colnames=names(churn.dat)
############################################################################
# Checking the summarized details of our data to get a basic idea about our dataset's attributes range.
summary(churn.dat)
dev.new()
par(mfrow=c(3,6))
for (i in 1:col){
  if(!is.factor(churn.dat[,i]))
    hist(churn.dat[,i], main= names(churn.dat[i]),col='pink')
}
dev.new()
par(mfrow=c(3,6))
for (i in 1:21){
  if(!is.factor(churn.dat[,i])){
    boxplot(churn.dat[,i]~class, main=names(churn.dat[i]), col= c('cyan','pink'))
  }
}

#Correlation matrix after removing the categorical columns
churn.corr = churn.dat[,-c(1,3,4,5,6,21)]
str(churn.corr)
str(churn.dat)
M=cor(churn.corr)
str(churn.corr)
dev.new()
corrplot(cor(churn.corr[sapply(churn.corr, is.numeric)]))


################### Removing variables that are highly correlated #################################
#Removing the following:
#total_day_charge
#total_eve_charge
#total_intl_charge
#total_night_charge
#Drop phone number also  from the analyses as it is unique and doesn't contribute much.

(iM = findCorrelation(M))
##################################################################################################
# Remodel the data frame by removing the corelated predictors
churn.remcor=churn.dat[,!names(churn.dat) %in% c("phone_number","total_day_charge","total_eve_charge",
                                              "total_intl_charge","total_night_charge")]
#churn.remcor=churn.dat[,-c(4,10,13,16,19)]
names(churn.remcor)
class.churn=churn.remcor$class
#Split the data into training and testing set
set.seed(101)
itrain=sample(nrow(churn.remcor), floor(0.7*row))
churn.remcor.train=churn.remcor[itrain,!names(churn.remcor) %in% c("class")]
churn.remcor.test=churn.remcor[-itrain,!names(churn.remcor) %in% c("class")]
churn.remcortr.class=class.churn[itrain]
churn.remcortest.class=class.churn[-itrain]
print(c(dim(churn.remcor.train),dim(churn.remcor.test),length(churn.remcortr.class),length(churn.remcortest.class)))
##################################################################################################

# Feature elimination (here based on random forests): Backward Selection by default
rfe.control=rfeControl(functions=rfFuncs,
                       method='cv',
                       number=10,
                       verbose=FALSE)
sizeset=c(5,10,15)
# Recursive Feature Elimination:
# (here based on a random forest)
rfe.out=rfe(churn.remcor.train,churn.remcortr.class,rfeControl = rfe.control,sizes=sizeset)
names(rfe.out)
print(rfe.out$bestSubset) # Best subset size of predictors
rfe.out$optVariables# Optimal Variables 
varImp(rfe.out)
##################################################################################################
#Also checking the important predictors using Logistic Regression
set.seed(101)
churn_glm=churn.remcor
str(x)
churn.glm<-glm(class~., data=churn_glm, family=binomial(link=logit))
summary(churn.glm)
dim(churn_glm)
###############################################################################################
## Remodel the data frame by removing the predictors that are not important as the 10 predictor set explains the
#data well. We are removing state also as the variable Importance score is very low, 1.859648 and the both feature
#techniques revealed either low scores or insignificant results.
churn.rfedat=churn.remcor[,!names(churn.remcor) %in% c("account_length", "area_code","total_day_calls"
                             ,"total_night_calls","total_eve_calls","state","class")]
names(churn.rfedat)

##################################################################################################
#Split the data into training and testing set to fit models to find the best model 
set.seed(101)
itrain=sample(nrow(churn.rfedat), floor(0.7*row))
churn.rfedat.train=churn.rfedat[itrain,]
churn.rfedat.test=churn.rfedat[-itrain,]
churn.train.class=class.churn[itrain]
churn.test.class=class.churn[-itrain]
print(c(dim(churn.rfedat.train),dim(churn.rfedat.test),length(churn.train.class),length(churn.test.class)))

#################################################################################################

# 4 different models are fitted using a 10 fold CV and different tuning parameters on the training data set 

#From above summary statistics, it shows us that all the attributes have a different range. So, we need to standardize 
#our data. We can standardize data using caret's preProcess() method.Also the predictors, number_vmail_messages, 
#total_intl_calls, number_customer_service_call are non-normal, as seen from the histogram, 
#we have scaled and centered the data before fitting the model for better analyses of the data.

######################## 1. LOGISTIC REGRESSION ###########################################
train.control=trainControl(method='cv',
                           number=10,
                           savePredictions = 'final',
                           verbose=FALSE,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary
                           )

par(mfrow=c(1,1))
set.seed(101)
churn.glm<-train(churn.rfedat.train,churn.train.class,
                 method='glm',family='binomial',
                 trControl=train.control,
                 preProcess=c('center','scale'),
                 metric='ROC')
summary(churn.glm)
churn.glm

glm.pred=predict(churn.glm,churn.rfedat.test)
confusionMatrix(churn.test.class, glm.pred)

glm.prob=predict(churn.glm,churn.rfedat.test,type='prob')

#######################################################################################
names(getModelInfo('knn'))
modelLookup('knn')
############################### 2. KNN Model ###########################################

train.control=trainControl(method='cv',
                           number=10,
                           savePredictions = 'final',
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary,
                           verbose=FALSE)

par(mfrow=c(1,1))
set.seed(101)
churn.knn<-train(churn.rfedat.train,churn.train.class,
                 method='knn',
                 trControl=train.control,
                 preProcess=c('center','scale'),
                 tuneLength = 4,
                 metric='ROC')
plot(churn.knn)
#plot(churn.knn$results$k,churn.knn$results$ROC, t='b',xlab= 'k neighbours', ylab='ROC')
knn.pred=predict(churn.knn,churn.rfedat.test)
confusionMatrix(churn.test.class,knn.pred)

knn.prob=predict(churn.knn,churn.rfedat.test,type='prob')

################################### 3. RANDOM FOREST ##########################################

train.control=trainControl(method='cv',
                           number=10,
                           verbose=FALSE,
                           savePredictions = 'final',
                           classProbs = TRUE,
                           summaryFunction= twoClassSummary)
set.seed(101)
churn.rf = train(churn.rfedat.train, churn.train.class,
               method= 'rf',
               trControl = train.control,
               metric='ROC',
               tuneLength=6)
plot(churn.rf)
varImp(churn.rf)
#plot(churn.rf$results$mtry,churn.rf$results$ROC,t='b', xlab= 'mtry', ylab='ROC')

rf.pred = predict(churn.rf,churn.rfedat.test)
confusionMatrix(churn.test.class,rf.pred)

rf.prob=predict(churn.rf,churn.rfedat.test,type='prob')

############################# 4. Support Vector Machine ###################################################
names(getModelInfo())
modelLookup('svmRadial')

train.control=trainControl(method='cv',
                           number=10,
                           savePredictions='final',
                           classProbs=TRUE,
                           summaryFunction=twoClassSummary
                           )
set.seed(101)
svm.train = data.frame(churn.rfedat.train, churn.train.class)
svm.out=train(churn.train.class~.,svm.train,
              method='svmRadial',
              metric='ROC',
              trControl=train.control,
              tuneLength= 8
              )
plot(svm.out)
svm.pred = predict(svm.out,churn.rfedat.test)
confusionMatrix(churn.test.class,svm.pred)

svm.prob=predict(svm.out,churn.rfedat.test,type='prob')

################################### ROC Curves ###########################################

roc(churn.test.class,glm.prob$NoChurn,plot=TRUE,col='red',print.auc=TRUE,print.auc.x=0.8)
roc(churn.test.class,knn.prob$NoChurn,plot=TRUE,col='blue',print.auc=TRUE,print.auc.y=0.8,add=TRUE)
roc(churn.test.class,rf.prob$NoChurn,plot=TRUE,col='black',add=TRUE, print.auc.y=0.9,print.auc=TRUE)
roc(churn.test.class,svm.prob$NoChurn,col='purple',plot=TRUE,add=TRUE, print.auc.y=1,print.auc=TRUE)
legend('bottomright', legend=c('Logistic','KNN','RF','SVM'),lty=1, col= c('red', 'blue', 'black','purple'))

########################### Predict the output using hold out set ###########################################
churn.holdout<-read.csv("C://Users//Roofiya Koya//Documents//Sem 2//CS6405//churn//churn-holdout.csv",header=TRUE)
head(churn.holdout)
attach(churn.dat)
holdout= churn.holdout[,!names(churn.holdout) %in% c("phone_number","total_day_charge","total_eve_charge",
                                                                  "total_intl_charge","total_night_charge",
                                                     "account_length", "area_code","total_day_calls"
                                                     ,"total_night_calls","total_eve_calls","state")]

churn.holdout$international_plan=as.factor(churn.holdout$international_plan)
churn.holdout$voice_mail_plan=as.factor(churn.holdout$voice_mail_plan)

######################### Predicting Churn on Test set using Random Forest#############################
pred=predict(churn.rf,churn.holdout)
Churn.out=ifelse(pred=='NoChurn',0,1)
table(Churn.out)

######################### Probability of Churn for Test set using Random Forest########################
rf.prob=predict(churn.rf,churn.holdout,type='prob')
predictedop= cbind(churn.holdout$phone_number,Churn.out,rf.prob)

######################### Writing output to CSV File##################################################
colnames(predictedop) = c('phone_number','pred','0','1')
table(pred)
write.csv(predictedop, file="Predictions.csv", row.names=FALSE)


