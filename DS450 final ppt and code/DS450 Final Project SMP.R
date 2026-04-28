# Importing and preliminary viewing of the data set
BikeRental<-read.csv(file="C:\\Users\\salpu\\Downloads\\hour.csv\\hour.csv",header=TRUE,sep=",")
head(BikeRental)
dim(BikeRental)
names(BikeRental)

#This checks what kind of data is in the columns, integers, numeric (continuous) or character variable (string)
str(BikeRental)

# Basic summary of the data for all the columns
summary(BikeRental)

# Checking for missing values
colSums(is.na(BikeRental))
# Note: Surprisingly, at least there are no values missing. 

hist(BikeRental$instant)
# Uniform

# Making a bar chart of average bike rentals per hour
avg_rentals <- aggregate(cnt ~ hr, data = BikeRental, FUN = mean)

# Create bar chart
barplot(
  avg_rentals$cnt,
  names.arg = avg_rentals$hr,
  xlab = "Hour of Day",
  ylab = "Average Number of Rentals",
  main = "Average Bike Rentals per Hour",
  col = "steelblue",
  las = 1
)

table(BikeRental$season)
# Provides the number of each season occurring in the data with 1 being spring and 4 being winter.

install.packages("corrplot")
library(corrplot)

corrplot(cor(BikeRental[sapply(BikeRental, is.numeric)]), method = "color")

# Let's run a forest model to attempt prediction
install.packages("randomForest")  
library(randomForest)

# These are to remove the two individual types of users renting bikes since we only care about the total.
BikeRental$casual <- NULL
BikeRental$registered <- NULL

# Converting categorical variables to ensure it works smoothly.
BikeRental$season <- as.factor(BikeRental$season)
BikeRental$yr <- as.factor(BikeRental$yr)
BikeRental$mnth <- as.factor(BikeRental$mnth)
BikeRental$hr <- as.factor(BikeRental$hr)
BikeRental$holiday <- as.factor(BikeRental$holiday)
BikeRental$weekday <- as.factor(BikeRental$weekday)
BikeRental$workingday <- as.factor(BikeRental$workingday)
BikeRental$weathersit <- as.factor(BikeRental$weathersit)

# Splitting the model into test and training sets
set.seed(123)

train_index <- sample(1:nrow(BikeRental), 0.8 * nrow(BikeRental))

train_data <- BikeRental[train_index, ]
test_data  <- BikeRental[-train_index, ]

# Creating the model
model_rf <- randomForest(
  cnt ~ ., 
  data = train_data,
  ntree = 200,      # number of trees
  mtry = 5,         # number of variables tried at each split
  importance = TRUE
)
predictions <- predict(model_rf, test_data)

# Evaluating the model with rmse and r2
rmse <- sqrt(mean((predictions - test_data$cnt)^2))
rmse

ss_total <- sum((test_data$cnt - mean(test_data$cnt))^2)
ss_res   <- sum((test_data$cnt - predictions)^2)

r2 <- 1 - (ss_res / ss_total)
r2
# r2 = 0.9357 which is ridiculously high, meaning it is quite a good predictor. 
# incMSE is VERY high, ~%150 for the hour variable in the random forest meaning hour is the most significant variable in determining rentals. 

# Now stating which variables are the most significant
importance(model_rf)
varImpPlot(model_rf)

# Now we do the gradient boosting model, start by importing necessary stuff.
install.packages("xgboost")   
install.packages("Matrix")    

library(xgboost)
library(Matrix)

# We already did most conversions before for the prior model, so now we convert data to a matrix.

# Combine first to ensure same structure
full_data <- rbind(train_data, test_data)

# Create matrix ONCE
full_matrix <- model.matrix(cnt ~ . - 1, data = full_data)

# Split back
train_matrix <- full_matrix[1:nrow(train_data), ]
test_matrix  <- full_matrix[(nrow(train_data) + 1):nrow(full_data), ]

# Labels
train_label <- train_data$cnt
test_label  <- test_data$cnt

# Construct the model
model_xgb <- xgboost(
  x = train_matrix,
  y = train_label,
  nrounds = 200,          # number of boosting rounds
  objective = "reg:squarederror",
  max_depth = 6,
  learning_rate = 0.1,              # learning rate
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 1
)

# Test the model
xgbpredictions <- predict(model_xgb, test_matrix)
xgbrmse <- sqrt(mean((xgbpredictions - test_label)^2))
xgbrmse

xgbss_total <- sum((test_label - mean(test_label))^2)
xgbss_res   <- sum((test_label - xgbpredictions)^2)

xgbr2 <- 1 - (xgbss_res / xgbss_total)
xgbr2

importance_matrix <- xgb.importance(model = model_xgb)
xgb.plot.importance(importance_matrix)
# The r2 value is less in this model than the prior model and that tracks because the rmse is higher, meaning more variation.
