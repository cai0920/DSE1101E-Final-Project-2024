rm(list = ls())
#dev.off()

#libraries
library(readr) # contains the dataset "bank-additional.csv"
library(ggplot2) # box plot 
library(ROCR) # ROC curve
library(kknn) # KNN for classification
library(e1071) # Naive Bayes classifier
library(tree) # Decision Tree 
library(rpart) # Decision Tree 
library(rpart.plot) # Decision Tree  
library(pls) #PCR

#set seed
set.seed(0301000) #student number

#read data
df = read_delim("bank-additional.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE)

#produce box plot of duration by y 
df_duration = df[c(11,21)]
ggplot(df_duration, aes(x = y, y = duration)) +
  geom_boxplot() +
  labs(title = "Box Plot of duration by y", x = "y", y = "duration") +
  theme_minimal()

#re-read data
df = read_delim("bank-additional.csv", delim = ";", escape_double = FALSE, 
                col_types = cols(duration = col_skip()), trim_ws = TRUE)
#duration highly affects the output target -> omit 
#note only one level illiterate in education, and one level yes in default. both led to y=no.
#remove these rows. 
which(grepl('illiterate', df$education))
which(grepl('yes', df$default))
df = df[-c(3927, 3515),]

#Rows: 4117 Columns: 20   

#convert character to factor 
cols_to_factor <- c(2:10,14,20)
df[cols_to_factor] <- lapply(df[cols_to_factor], factor)

#delete missing values (not used)
num_rows_with_unknown <- sum(rowSums(df == "unknown", na.rm = TRUE) > 0)
num_rows_with_unknown
#rows removed = 1029 -> 25% of data removed -> too much information lost 

#check if missing values are treated as a possible class label
levels(df$housing)
#"no"      "unknown" "yes"   

#correlation of social and economic context attributes
df_socialeconomic = df[c(15:19)]
# Calculate the correlation matrix
cor_matrix <- cor(df_socialeconomic)
cor_matrix
#               emp.var.rate cons.price.idx cons.conf.idx euribor3m nr.employed
#emp.var.rate      1.0000000      0.7549262     0.1961142 0.9703181   0.8971960
#cons.price.idx    0.7549262      1.0000000     0.0469855 0.6569788   0.4723596
#cons.conf.idx     0.1961142      0.0469855     1.0000000 0.2776190   0.1077497
#euribor3m         0.9703181      0.6569788     0.2776190 1.0000000   0.9425778
#nr.employed       0.8971960      0.4723596     0.1077497 0.9425778   1.0000000

#emp.var.rate, cons.price.idx, euribor3m, nr.employed are moderately/highly correlated 


###############################
###Principal Component Analysis
###############################

#remove cons.conf.idx
df_socialeconomic = df_socialeconomic[c(1,2,4,5)]

#Perform the PCA decomposition
prall = prcomp(df_socialeconomic, scale = TRUE)

#Biplot
biplot(prall)

#summary
prall.s = summary(prall)

scree = prall.s$importance[2,] #save the proportion of variance explained

#Finally, do the scree plot:
plot(scree, main = "Scree Plot", xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", ylim = c(0,1), type = 'b', cex = .8)

sum(prall.s$importance[2,][1:2])
#0.98612
#PC1 and PC2 explains 98.6% of the variance in the data. 

# Extract the first 2 principal components
pcs = prall$x[, 1:2]

#bind PCs
df1 = cbind(df, pcs)
df1 = df1[c(1:14,17,20:22)] #delete original inputs 
#df = df1 run this to compare auc 

###################
###Logit Regression
###################

ntrain=2889#~70/30 split

tr = sample(1:nrow(df),ntrain) 
train = df[tr,]  
test = df[-tr,]  

#Fit the logit model on training data:
glm_fit = glm(y~., data = train, family = "binomial")

#Backward selection -> run this to do backward selection 
#glm_fit = glm(y ~ ., data = train, family = "binomial")
#step(glm_fit, direction = "backward")

#glm_fit = glm(formula = y ~ age + contact + month + campaign + previous + 
#poutcome + cons.conf.idx + PC1 + PC2, family = "binomial", data = train) #best fit after pca 

#glm_fit = glm(formula = y ~ age + contact + month + campaign + poutcome + 
#emp.var.rate + cons.price.idx + cons.conf.idx + euribor3m, family = "binomial", data = train) #best fit without pca 

#Predict the test observations using the trained logit model in glm_train:
glm_prob = predict(glm_fit, newdata = test, type = "response")

#Build the confusion matrix:
table(glm_prob > 0.5, test$y)
#PCA with backward selection 
#        no  yes
#FALSE 1081  104
#TRUE    15   28
#sensitivity = 0.986

#no PCA with backward selection and PCA with no backward selection
#        no  yes
#FALSE 1075  104
#TRUE    21   28
#sensitivity = 0.981

#no PCA no backward selection 
#        no  yes
#FALSE 1071  106
#TRUE    25   26
#sensitivity = 0.978

#Compute and plot the ROC curve:
pred = prediction(glm_prob, test$y)
perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc_perf = performance(pred, measure = "auc") # Calculate AUC
plot(perf, col = "steelblue", lwd = 2) # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.8, paste("AUC =", round(auc_perf@y.values[[1]], 2))) # Compute AUC and add text to ROC plot.
#AUC = 0.76 for all

###############################
###K nearest neighbours & LOOCV
###############################

#remove all nominal categorical input variables! df[c(2,3,5:10,15)]

df.loocv=train.kknn(y~.-job-marital-default-housing-loan-contact-month-day_of_week-poutcome, data=train,kmax=100, kernel = "rectangular")

#Examine the behavior of the LOOCV misclassification:
plot((1:100),df.loocv$MISCLASS, type="l", col = "blue", main="LOOCV Misclassification", xlab="Choice of K", ylab="Misclassification rate")

#Find the best K:
kbest=df.loocv$best.parameters$k

#Now calculate the predictions with optimal K:
knnpredcv=kknn(y~.-job-marital-default-housing-loan-contact-month-day_of_week-poutcome,train,test,k=kbest,kernel = "rectangular")

#Build the confusion matrix
table(knnpredcv$fitted.values, test$y) 
#no PCA 
#      no  yes
#no  1080  103
#yes   16   29

#PCA 
#      no  yes
#no  1082  106
#yes   14   26

pred = prediction(knnpredcv$prob[,2], test$y)
perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc_perf = performance(pred, measure = "auc") # Calculate AUC
plot(perf, col = "steelblue", lwd = 2) # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.8, paste("AUC =", round(auc_perf@y.values[[1]], 2))) # Compute AUC and add text to ROC plot.
#no PCA: AUC = 0.74, worse than logit 
#PCA: AUC = 0.73 

#########################
###Naïve Bayes Classifier 
#########################

#We first train the classifier on the training data:
nbfit<-naiveBayes(y~., data=train)

#Predict class based on test data - for confusion matrix:
nbpred=predict(nbfit, test, type="class")

#Predict probabilities on the test data - for the ROC curve:
nbpred2=predict(nbfit, test, type="raw")

#Produce the confusion matrix
table(nbpred, test$y)
#no PCA 
#nbpred  no  yes
#    no  977  72
#    yes 119  60

#PCA 
#nbpred  no  yes
#    no  1032 85
#    yes 64   47

#Produce the ROC curve (use the second column of nbpred2, as it contains estimated P(Y=1|X)
pred = prediction(nbpred2[,2], test$y)
perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc_perf = performance(pred, measure = "auc") # Calculate AUC
plot(perf, col = "steelblue", lwd = 2) # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.8, paste("AUC =", round(auc_perf@y.values[[1]], 2))) # Compute AUC and add text to ROC plot.
#no PCA: AUC = 0.77, better than logit 
#PCA: AUC = 0.76 

#################
###Decision Trees 
#################

#First, we fit a big tree using the Gini loss (default in rpart)
treeGini= rpart(y~., data=train, method = "class", minsplit = 20, cp = 0.001, maxdepth = 30)

#Examine the cross-validation plot:
plotcp(treeGini)

#Find the best complexity parameter on CV:
bestcp=treeGini$cptable[which.min(treeGini$cptable[,"xerror"]),"CP"]
bestcp
#bestcp around 0.125

#Retrieve xerr -> verify that the results are stable, keep the one with the lowest/most occuring xerr by running the code again
bestxerr=treeGini$cptable[which.min(treeGini$cptable[,"xerror"]),"xerror"] 
bestxerr
#bestxerr around 0.884

#Prune the tree according to the chosen CP:
bestGini = prune(treeGini,cp=bestcp)

#Check the leaves
length(unique(bestGini$where))
#3 no PCA 
#5 with PCA 

#Predict on test data 
treepred = predict(bestGini,newdata = test) #predict the test observations

#ROC curve
pred = prediction(treepred[,2], test$y)
perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc_perf = performance(pred, measure = "auc") # Calculate AUC
plot(perf, col = "steelblue", lwd = 2, main="ROC for Tree") # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.8, paste("AUC =", round(auc_perf@y.values[[1]], 2))) # Compute AUC and add text to ROC plot.
#AUC = 0.68, worse than Naïve Bayes, same with PCA & without PCA 

#plot best tree
rpart.plot(bestGini, shadow.col = "gray")
