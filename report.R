data <- read.csv("https://www.maths.dur.ac.uk/users/hailiang.du/assignment_data/hotels.csv")

data <- data[, !(names(data) %in% c("reservation_status", "reservation_status_date"))]
data$agent[data$agent == "NULL"] <- NA
data$company[data$company == "NULL"] <- NA

data <- data[, !(names(data) %in% c("agent", "company", "country"))]

data <- na.omit(data)

char_cols <- sapply(data, is.character)
data[, char_cols] <- lapply(data[, char_cols], as.factor)

data$is_canceled <- as.factor(data$is_canceled)

library(caret)

set.seed(123)
train_index <- createDataPartition(data$is_canceled, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

model_glm <- glm(
  is_canceled ~ .,
  data = train_data,
  family = binomial()
)

prob_glm <- predict(model_glm, newdata = test_data, type = "response")
pred_glm <- ifelse(prob_glm > 0.5, "1", "0")
pred_glm <- factor(pred_glm, levels = c("0", "1"))

confusionMatrix(pred_glm, test_data$is_canceled)

library(rpart)
library(rpart.plot)

model_tree <- rpart(
  is_canceled ~ .,
  data = train_data,
  method = "class"
)

pred_tree <- predict(model_tree, test_data, type = "class")

confusionMatrix(pred_tree, test_data$is_canceled)

library(pROC)

prob_glm <- predict(model_glm, newdata = test_data, type = "response")

roc_glm <- roc(test_data$is_canceled, prob_glm)

plot(roc_glm, main = "ROC Curve - Logistic Regression")

auc(roc_glm)

rpart.plot(model_tree)

library(caret)

ctrl <- trainControl(method = "cv", number = 5)

grid <- expand.grid(cp = seq(0.001, 0.02, by = 0.002))

model_tree_tuned <- train(
  is_canceled ~ .,
  data = train_data,
  method = "rpart",
  trControl = ctrl,
  tuneGrid = grid
)

print(model_tree_tuned)

pred_tree_tuned <- predict(model_tree_tuned, test_data)

confusionMatrix(pred_tree_tuned, test_data$is_canceled)

library(ggplot2)

ggplot(data, aes(x = lead_time, fill = is_canceled)) +
  geom_histogram(bins = 50, position = "identity", alpha = 0.6) +
  labs(title = "Lead Time vs Cancellation")

cm <- confusionMatrix(pred_glm, test_data$is_canceled)
cm

precision <- cm$byClass["Pos Pred Value"]
recall <- cm$byClass["Sensitivity"]
f1 <- 2 * (precision * recall) / (precision + recall)

precision
recall
f1

prob_glm <- predict(model_glm, newdata = test_data, type = "response")

thresholds <- c(0.3, 0.4, 0.5, 0.6)

results <- data.frame(
  threshold = numeric(),
  accuracy = numeric(),
  precision = numeric(),
  recall = numeric(),
  f1 = numeric()
)

for (th in thresholds) {
  pred <- ifelse(prob_glm > th, "1", "0")
  pred <- factor(pred, levels = c("0", "1"))
  
  cm <- confusionMatrix(pred, test_data$is_canceled, positive = "1")
  
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  results <- rbind(results, data.frame(
    threshold = th,
    accuracy = cm$overall["Accuracy"],
    precision = precision,
    recall = recall,
    f1 = f1
  ))
}

results

calibration_df <- data.frame(
  prob = prob_glm,
  actual = as.numeric(as.character(test_data$is_canceled))
)

calibration_df$bin <- cut(calibration_df$prob, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)

calibration_summary <- aggregate(cbind(prob, actual) ~ bin, data = calibration_df, mean)
calibration_summary

plot(
  calibration_summary$prob,
  calibration_summary$actual,
  xlab = "Mean Predicted Probability",
  ylab = "Observed Cancellation Rate",
  main = "Calibration Plot"
)
abline(0, 1, lty = 2)
 
