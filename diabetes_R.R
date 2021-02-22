# We'll use the following 2 algorithms for the prediction of diabetic in pregnant women:
# 1. Decision Tree;
# 2. Support Vector Machine (SVM) and;
# 3. Comparison of Model Accuracy.

library(corrplot)
library(caret)


dataset <- read.csv("diabetes.csv", col.names=c("Pregnant","Plasma_Glucose","Dias_BP","Triceps_Skin","Serum_Insulin","BMI","DPF","Age","Diabetes"))

# Print the dimensions of dataset
dim(dataset)

# shows the structure of the dataset
str(dataset) 

# Print the top 5 rows of the dataset
head(dataset)

# Print the last 5 rows of the dataset
#tail(dataset)

# Print a quick summary of the dataset
summary(dataset)

# “sapply”" to check the number of missing values in each columns.
sapply(dataset, function(x) sum(is.na(x)))

# Since there are not missing values on the data let’s produce the matrix of scatterplots.
pairs(dataset, panel = panel.smooth)

# we compute the matrix of correlations between the variables
corrplot(cor(dataset[, -9]), type = "upper", method = "number")


# 1. Decision Trees algorithm:

# Preparing the data:
dataset$Diabetes <- as.factor(dataset$Diabetes)

library(caret)
library(tree)
library(e1071)

set.seed(1000)
intrain <- createDataPartition(y = dataset$Diabetes, p = 0.8, list = FALSE)
train <- dataset[intrain, ]
test <- dataset[-intrain, ]

# Training The Model
treemod <- tree(Diabetes ~ ., data = train)
summary(treemod)

# The summary show that were used 5 variables are internal nodes in the tree, 
# 10 terminal nodes and the training error rate is 22.11%.

# Let's getet a detailed text output.
treemod

# The results display the split criterion (e.g. Plasma_Glucose < 127.5), 
# the number of observations in that branch, the deviance, the overall 
# prediction for the branch (Yes or No), and the fraction of observations
# in that branch that take on values of Yes and No. Branches that lead 
# to terminal nodes are indicated using asterisks.

# Let's plot the tree, and see the results.
plot(treemod)
text(treemod, pretty = 0)

# “Diabetes” appears to be "Plasma_Glucose", since the first branch split criterion (e.g. Plasma_Glucose < 127.5).

# Predict the response on the test data, and produce a confusion matrix comparing 
# the test labels to the predicted test labels. Then we will see the test error rate?

# Testing the Model
tree_pred <- predict(treemod, newdata = test, type = "class" )
confusionMatrix(tree_pred, test$Diabetes)

acc_treemod <- confusionMatrix(tree_pred, test$Diabetes)$overall['Accuracy']
print(acc_treemod)

# The test error rate is 23.5%. In other words, the accuracy is 76.5%.
# These numbers may slightly differ once we re-train the models

# Let's applying Support Vector Machine - svm model
#Preparing the DataSet:
set.seed(1000)
intrain <- createDataPartition(y = dataset$Diabetes, p = 0.8, list = FALSE)
train <- dataset[intrain, ]
test <- dataset[-intrain, ]

# Let's choose the parameters: Now, we will use the tune() function to do a grid   
# search over the supplied parameter ranges (C - cost, gamma), using the train set. 
# The range to gamma parameter is between 0.000001 and 0.1. 
# For cost parameter the range is from 0.1 until 10. 
# It’s important to understanding the influence of this two parameters, 
# because the accuracy of an SVM model is largely dependent on the selection them.

tuned <- tune.svm(Diabetes ~., data = train, gamma = 10^(-6:-1), cost = 10^(-1:1))
summary(tuned) 

# The best parameters are Cost=1 and gamma=0.01.
# In order to build a svm model to predict “Diabetes” using Cost=1 and gamma=0.01, 


svm_model  <- svm(Diabetes ~., data = train, kernel = "radial", gamma = 0.01, cost = 10) 
summary(svm_model)

# Now let's test the model:
svm_pred <- predict(svm_model, newdata = test)
confusionMatrix(svm_pred, test$Diabetes)

acc_svm_model <- confusionMatrix(svm_pred, test$Diabetes)$overall['Accuracy']
# The test error rate is 21.9%. In other words, the accuracy is 79.1%.
# These numbers may slightly differ once we re-train the models

# Comparison of two models
accuracy <- data.frame(Model=c("Decision Tree", "Support Vector Machine (SVM)"), Accuracy=c(acc_treemod, acc_svm_model ))
ggplot(accuracy,aes(x=Model,y=Accuracy)) + geom_bar(stat='identity') + theme_bw() + ggtitle('Comparison of Model Accuracy')
