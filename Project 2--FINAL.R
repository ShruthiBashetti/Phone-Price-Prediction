#getwd()
setwd('C:/Users/Shruthi Bashetti/Desktop/Practicum/Project 2')
##Phone Dataset Has diffrent attributes such as RAM,Battery, Used price etc
##This data is useful to build models that predict prices of used phones.

##libraries

library(magrittr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(Boruta)
library(rpart)
library(rpart.plot)
library(fastDummies)
library(neuralnet)
library(nnet)
library(caret)
library(e1071)
library(class)

##Load the dataset
phone <- read.csv('used_device_data.csv')
head(phone)
str(phone)
dim(phone)
names(phone)

summary(phone)
summary(phone$normalized_used_price)
summary(phone$normalized_new_price)

##unique values
unique(phone$device_brand)
sort(table(phone$device_brand), decreasing = TRUE)
unique(phone$os)
sort(table(phone$os), decreasing = TRUE)
unique(phone$internal_memory)
unique(phone$X4g)
table(phone$X4g)
unique(phone$X5g)
table(phone$X5g)


##checking missing values
sum(is.na(phone))
colSums(is.na(phone)) ##rear_camera has 179 missing values


replace_missing_with_median <- function(x) {
  median_value <- median(x, na.rm = TRUE)
  ifelse(is.na(x), median_value, x)
}


# Apply the custom function within each brand group for each column
columns_to_process <- c('rear_camera_mp',"front_camera_mp", "internal_memory", "ram", "battery", 
                        "weight")

phone <- phone %>%
  group_by(device_brand,os) %>%
  mutate(across(all_of(columns_to_process), replace_missing_with_median)) %>%
  ungroup()

colSums(is.na(phone))

##checking 10 values of rear-end
records_with_na <- phone %>% filter(is.na(rear_camera_mp))
records_with_na

##checking device = infinix
infinix_records <- phone %>% filter(device_brand == "Infinix")
dim(infinix_records)

##
#replace infinix others with 0 for rear camera
phone <- phone %>%
  mutate(rear_camera_mp = ifelse(device_brand == "Infinix", replace(rear_camera_mp, is.na(rear_camera_mp), 0), rear_camera_mp))

##

phone %>% filter(is.na(battery))
phone %>% filter(is.na(ram))
phone %>% filter(is.na(weight))


##checking for 0's
any(phone == 0)
records_with_zeros <- phone %>%
  filter(rowSums(phone == 0) > 0)
records_with_zeros ##These are all nokia with others on os.



##variance
var(phone$release_year)
str(phone)
numeric_columns <- phone[sapply(phone, is.numeric)]
# Calculate the variance for each numeric column
variance_values <- apply(numeric_columns, 2, var)
# Print the result
print(variance_values)


####HANDLING DATATYPES
#phone$X4g <- as.integer(ifelse(phone$X4g == 'yes', 1, 0))
#str(phone$X4g)
#phone$X5g <- as.integer(ifelse(phone$X5g == 'yes', 1, 0))

#phone$X4g <- factor(phone$X4g, levels = c('no','yes'), labels = c(0,1))
#phone$X5g <- factor(phone$X5g, levels = c('no','yes'), labels = c(0,1))
str(phone$X5g)


###exploration
names(phone)
ggplot(phone, aes(y = device_brand)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Count Plot for Device Brand",
       x = "Device Brand",
       y = "Count")

ggplot(phone, aes(x = os)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Bar Plot for OS",
       x = "Device Brand",
       y = "Count")

ggplot(phone, aes(x = X4g)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Bar Plot for 4G",
       x = "4G",
       y = "Count")


ggplot(phone, aes(x = X5g)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Bar Plot for 5G",
       x = "5G",
       y = "Count")

ggplot(phone, aes(x = release_year)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Bar Plot for 5G",
       x = "Device Brand",
       y = "Count")


hist(phone$normalized_used_price,main = "Frequency Distribution of used phone prices", col = 'skyblue')
hist(phone$normalized_new_price,main = "Frequency Distribution of new phone prices", col = 'skyblue')
hist(phone$screen_size,main = "Frequency Distribution of screensizes", col = 'skyblue')
hist(phone$internal_memory,main = "Frequency Distribution of internal memory", col = 'skyblue')
hist(phone$ram,main = "Frequency Distribution of ram", col = 'skyblue')
hist(phone$battery,main = "Frequency Distribution of battery", col = 'skyblue')
hist(phone$weight,main = "Frequency Distribution of weight", col = 'skyblue')
hist(phone$days_used,main = "Frequency Distribution of days used", col = 'skyblue')

plot(phone$normalized_new_price ,phone$normalized_used_price,
     xlab = "Normalized New Price", ylab = "Normalized Used Price",
     main = "Scatter Plot of Normalized Prices",col='blue')


boxplot(phone$screen_size,main = "Boxplot Distribution of screensizes", col = 'skyblue')
boxplot(phone$internal_memory,main = "Boxplot Distribution of internal memory", col = 'skyblue')
boxplot(phone$ram,main = "Boxplot Distribution of ram", col = 'skyblue')
boxplot(phone$battery,main = "Boxplot Distribution of battery", col = 'skyblue')
boxplot(phone$weight,main = "Boxplot Distribution of weight", col = 'skyblue')
boxplot(phone$days_used,main = "Boxplot Distribution of days used", col = 'skyblue')

###################OUTLIERS###########################
find_outliers_IQR <- function(feature, phone) {
  q1 <- quantile(phone[[feature]], 0.25)
  q3 <- quantile(phone[[feature]], 0.75)
  
  IQR <- q3 - q1
  
  outliers <- phone[(phone[[feature]] < (q1 - 2.5 * IQR)) | (phone[[feature]] > (q3 + 2.5 * IQR)), ]
  
  return(outliers[order(outliers[[feature]]), ])
}

find_outliers_IQR("rear_camera_mp", phone)
find_outliers_IQR("weight", phone)
summary(phone$weight)
unique(phone$weight)
hist(phone$weight)
phone$weight <- log(phone$weight)

#############predictor Relevancy#####################
numeric_columns <- phone[sapply(phone, is.numeric)]
names(numeric_columns)

cor_matrix <- cor(numeric_columns) 


cor_matrix_melted <- melt(cor_matrix)

# Create a heat map using ggplot2
ggplot(data = cor_matrix_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2( midpoint = 0, limit = c(-1,1), space = "Lab",
                        name="Correlation") +
  geom_text(aes(x = Var1, y = Var2, label = round(value, 2)), 
            color = "black", size = 4) +  # Add correlation values
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 10, hjust = 1)) +
  coord_fixed()


str(phone)


####################FEATURE ENGINEERING##################
names(phone)



# Define function to map values to class
map_to_class <- function(value, class_ranges) {
  for (i in 1:length(class_ranges)) {
    if (value <= class_ranges[[i]]$max) {
      return(class_ranges[[i]]$class)
    }
  }
}

# Define class ranges
ram_classes <- list(
  list(class = "Low", max = 4),
  list(class = "Medium", max = 8),
  list(class = "High", max = Inf)
)

internal_memory_classes <- list(
  list(class = "Low", max = 24),
  list(class = "Medium", max = 48),
  list(class = "High", max = Inf)
)

screen_size_classes <- list(
  list(class = "Low", max = 10),
  list(class = "Medium", max = 20),
  list(class = "High", max = Inf)
)


battery_capacity_classes <- list(
  list(class = "Low", max = 4000),
  list(class = "Medium", max = 8000),
  list(class = "High", max = Inf)
)


# Apply function to each relevant column in the DataFrame
phone <- phone %>%
  mutate(
    RAM_Class = sapply(ram, function(x) map_to_class(x, ram_classes)),
    Internal_Memory_Class = sapply(internal_memory, function(x) map_to_class(x, internal_memory_classes)),
    Screen_Size_Class = sapply(screen_size, function(x) map_to_class(x, screen_size_classes)),
    Battery_Capacity_Class = sapply(battery, function(x) map_to_class(x, battery_capacity_classes)),
  )

# Define weights for each feature
weights <- c(RAM = 0.25, Internal_Memory = 0.25, Screen_Size = 0.25, Battery_Capacity = 0.25 )
phone$Weighted_Avg <- apply(phone[, c("RAM_Class", "Internal_Memory_Class", "Screen_Size_Class", "Battery_Capacity_Class")], 1, function(x) sum(match(x, c("Low", "Medium", "High")) * weights) / sum(weights))
# Define thresholds to classify the weighted average into low, medium, or high
threshold_low <- 1.3
threshold_high <- 1.6
# Determine the final class based on the weighted average
phone$class <- ifelse(phone$Weighted_Avg < threshold_low, "Low", 
                      ifelse(phone$Weighted_Avg < threshold_high, "Medium", "High"))

table(phone$class)

##DROP RAM<, MEMORY CLASS, SCREENSIZE,BATTERY, WEIGHT_AVG
names(phone)
phone <- phone[,-c(16:20)]


##################DImension Reduction####################
names(phone)
#please run these codes, which take few mins to execute
#boruta_results <- Boruta(normalized_used_price ~., data = phone[,-16])
#boruta_results

#boruta_results_class <- Boruta(as.factor(class) ~., data = phone[,-14])
#boruta_results_class



##########Data Partitioning###############
summary(phone)
str(phone)

##partitioning the data into 50:25:25
set.seed(123)
train_idx <- caret::createDataPartition(phone$normalized_used_price, p = 0.5, list = FALSE)

# Subset the remaining 60% for validation and test sets
remaining_data <- phone[-train_idx, ]
dim(remaining_data)
# Create data partition with 60% for validation and 40% for test
validation_idx <- caret::createDataPartition(remaining_data$normalized_used_price, p = 0.50, list = FALSE)

# Assign data to train, validation, and test sets
phone_train <- phone[train_idx, ]
phone_valid <- remaining_data[validation_idx, ]
phone_holdout <- remaining_data[-validation_idx, ]

dim(phone_train)
dim(phone_valid)
dim(phone_holdout)

###############################################################################
#####*********Model1****FEATURE SELECTION*******Predicting Price************#############
names(phone_train)
model1_price <- lm(normalized_used_price~. , data = phone_train[,-16]) #or 16 for class
summary(model1_price)

#####Feature Selection##############
model1_price_feature <- step(model1_price, direction = "backward")
summary(model1_price_feature)

##model 1
regression_price <- lm(normalized_used_price~ screen_size + X4g + X5g + rear_camera_mp +front_camera_mp+ram+battery+weight+release_year+normalized_new_price, data = phone_train[,-16]) #or 16 
summary(model1_price)

predicted_price_regmodel <- predict(regression_price,newdata = phone_valid)
summary(predicted_price_regmodel)
summary(phone_valid$normalized_used_price)
cor(predicted_price_regmodel, phone_valid$normalized_used_price)

plot(predicted_price_regmodel, phone_valid$normalized_used_price)

forecast::accuracy(predicted_price_regmodel, phone_valid$normalized_used_price)


##Model2
names(phone)
used_price_rpart<-rpart(normalized_used_price~., data=phone_train[,-16]) ##16 for class
used_price_rpart

rpart.plot(used_price_rpart, digits=3)


#fancyRpartPlot(used_price_rpart, cex = 0.8)
predictedprice_tree<-predict(used_price_rpart, phone_valid)
summary(predictedprice_tree)
summary(phone_valid$normalized_used_price)
cor(predictedprice_tree, phone_valid$normalized_used_price)

##Evaluation

forecast::accuracy(predictedprice_tree, phone_valid$normalized_used_price)

##Model3 Nueral Networks

phone1 <- dummy_cols(phone, select_columns = c("os",'device_brand','X4g','X5g'), remove_selected_columns = TRUE)
names(phone1)
set.seed(123)
train_idx <- caret::createDataPartition(phone1$normalized_used_price, p = 0.5, list = FALSE)

# Subset the remaining 60% for validation and test sets
remaining_data <- phone1[-train_idx, ]
dim(remaining_data)
# Create data partition with 60% for validation and 40% for test
validation_idx <- caret::createDataPartition(remaining_data$normalized_used_price, p = 0.5, list = FALSE)

# Assign data to train, validation, and test sets
phone1_train <- phone1[train_idx, ]
phone1_valid <- remaining_data[validation_idx, ]
phone1_holdout <- remaining_data[-validation_idx, ]

dim(phone1_train)
dim(phone1_valid)
dim(phone1_holdout)

normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
names(phone1_train)
str(phone1_train)
phone_norm_train <-as.data.frame(lapply(phone1_train[,-12], normalize)) ##i dont have class in this dataframe
phone_norm_valid <-as.data.frame(lapply(phone1_valid[,-12], normalize)) ##i dont have class in this dataframe

phoneprice_model<-neuralnet(normalized_used_price ~ ., data=phone_norm_train,hidden=1) #c(,1:11,13:51)

plot(phoneprice_model) 

phoneprice_pred<-compute(phoneprice_model, phone_norm_valid)
predicted_price_nn <-phoneprice_pred$net.result

cor(predicted_price_nn, phone_norm_valid$normalized_used_price)

unnormalize <- function(x, minv, maxv) {
  return( x * (maxv - minv) + minv )
}

unnorm_phoneprice <- unnormalize(predicted_price_nn, min(phone1_valid$normalized_used_price), max(phone1_valid$normalized_used_price))

cor(unnorm_phoneprice, phone1_valid$normalized_used_price)


forecast::accuracy(phone1_valid$normalized_used_price, unnorm_phoneprice)

##############HOLDOUTS WITH SELECTED MODEL-LINEAR REGRESSION##########
dim(phone_holdout)
predicted_price_regmodel <- predict(regression_price,newdata = phone_holdout)
summary(predicted_price_regmodel)
summary(phone_valid$normalized_used_price)
cor(predicted_price_regmodel, phone_holdout$normalized_used_price)

plot(predicted_price_regmodel, phone_holdout$normalized_used_price)

forecast::accuracy(predicted_price_regmodel, phone_holdout$normalized_used_price)





###############################################################################
#####*****Model2*****FEATURE SELECTION*****Classifying the phone****#############
##data partitioning for classifying the phone
str(phone)
phone1 <- dummy_cols(phone, select_columns = c("os",'device_brand','X4g','X5g'), remove_selected_columns = TRUE)
names(phone1)

set.seed(2027)
train_idx <- caret::createDataPartition(phone1$class, p = 0.6, list = FALSE)

# Subset the remaining 60% for validation and test sets
remaining_data <- phone1[-train_idx, ]
dim(remaining_data)
# Create data partition with 60% for validation and 40% for test
validation_idx <- caret::createDataPartition(remaining_data$class, p = 0.50, list = FALSE)

# Assign data to train, validation, and test sets
phone_train <- phone1[train_idx, ]
phone_valid <- remaining_data[validation_idx, ]
phone_holdout <- remaining_data[-validation_idx, ]

dim(phone_train)
dim(phone_valid)
dim(phone_holdout)

unique(phone$class)
names(phone_train)

##model1
model2_class <- multinom(class ~ ., data = phone_train[,-10]) #or 14 used price
summary(model2_class)

model_class_prediction <- predict(model2_class,phone_valid[,-10], type="probs")
model_class_prediction

predicted_class <- apply(model_class_prediction, 1, which.max)

# Map numerical class labels to actual class names
class_names <- c("High", "Medium", "Low")
predicted_class_names <- class_names[predicted_class]

# Display the predicted class names
print(predicted_class_names)

conf_matrix <- confusionMatrix(table(predicted_class_names, phone_valid$class))
conf_matrix






#####Classification Tree
names(phone_train)
class_tree_model <- rpart(class ~ ., data = phone_train[,-10],method = "class", minbucket = 2, maxdepth = 5)

class_tree_model
t(t(class_tree_model$variable.importance))
#printcp(class_tree_model)
#plotcp(class_tree_model)
prp(class_tree_model, type = 2, extra = 104, nn = TRUE, fallen.leaves = TRUE, faclen = 4, varlen = 8, shadow.col = "skyblue")


class_tree_model

probabilities <-predict(class_tree_model, phone_valid[ ,-10]) ##purchase, predicted purchase from regression

threshold <- 0.5
decide_class <- function(probabilities, threshold) {
  classes <- colnames(probabilities)
  decided_class <- apply(probabilities, 1, function(row) {
    if (max(row) >= threshold) {
      return(classes[which.max(row)])
    } else {
      return("Undecided")  
    }
  })
  return(decided_class)
}

# Apply the function to decide the class
predicted_class <- decide_class(probabilities, threshold)

# Display the predicted class
print(predicted_class)
unique(predicted_class)


conf_matrix <- confusionMatrix(data = factor(predicted_class, levels = c("High", "Medium", "Low")),
                               reference = factor(phone_valid$class, levels = c("High", "Medium", "Low")))

# Display the confusion matrix
print(conf_matrix)

#####NaiveBayes

phoneclass.nb <- naiveBayes(class ~ ., data = phone_train[,-10])
phoneclass.nb
phoneclass_pred.prob <- predict(phoneclass.nb, newdata = phone_valid[,-10], type = "raw")
phoneclass_pred.class <- predict(phoneclass.nb, newdata = phone_valid[,-10])
phoneclass_pred.class
#library(caret)
conf_matrix <- confusionMatrix(data = factor(phoneclass_pred.class, levels = c("High", "Medium", "Low")),
                               reference = factor(phone_valid$class, levels = c("High", "Medium", "Low")))

conf_matrix


##KNN
k <- 3  # Choose the number of neighbors
names(phone_train)
knn_model <- knn(train = phone_train[,-c(10,12)], test = phone_valid[,-c(10,12)], cl = phone_train$class, k = k)

conf_matrix <- confusionMatrix(data = factor(knn_model, levels = c("High", "Medium", "Low")),
                               reference = factor(phone_valid$class, levels = c("High", "Medium", "Low")))
conf_matrix

str(phone_train)

y_train <- as.factor(phone_train$class)
knntuning = tune.knn(x= phone_train[,-c(10,12)], y = y_train, k = 1:30)
knntuning
plot(knntuning)

summary(knntuning)

k <- 1  # Best k
names(phone_train)
knn_model <- knn(train = phone_train[,-c(10,12)], test = phone_valid[,-c(10,12)], cl = phone_train$class, k = k)

conf_matrix <- confusionMatrix(data = factor(knn_model, levels = c("High", "Medium", "Low")),
                               reference = factor(phone_valid$class, levels = c("High", "Medium", "Low")))
conf_matrix



##holdouts 
phone_holdout
names(phone_holdout)

k=1
knn_model_final_holdouts <- knn(train = phone_train[,-c(10,12)], test = phone_holdout[,-c(10,12)], cl = phone_train$class, k = k)

conf_matrix <- confusionMatrix(data = factor(knn_model, levels = c("High", "Medium", "Low")),
                               reference = factor(phone_valid$class, levels = c("High", "Medium", "Low")))
conf_matrix









