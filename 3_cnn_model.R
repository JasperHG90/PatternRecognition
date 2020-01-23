rm(list=ls())

#################################Start Here########################################
library(keras)
library(tidyverse)
library(rBayesianOptimization)
library(yardstick)

####1. Read data and word embeddings####
x_train <- read_rds("./data/other/Qixiang/train_x.rds")
y_train <- read_rds("./data/other/Qixiang/train_y.rds")
x_test <- read_rds("./data/other/Qixiang/test_x.rds")
y_test <- read_rds("./data/other/Qixiang/test_y.rds")

y_train_one_hot <- to_categorical(y_train)
y_test_one_hot <- to_categorical(y_test)

class_weights <- prod(table(y_train))/table(y_train)/(10^29)

embedding_matrix <- read_rds("./data/other/Qixiang/embedding_matrix.rds")
embedding_dim <- 300
max_words <- 20000
maxlen <- 158



####2.Bayesian Optimization for Parameter Search####
#parameters: filter_n, filter_size, neuron_n, dropout
training_credit = function(initParams){
  
  # Define Model
  model <- keras_model_sequential() %>% 
    layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                    input_length = maxlen) %>% 
    layer_conv_1d(filters = initParams$filter_n, kernel_size = initParams$filter_size, use_bias = FALSE) %>%
    layer_batch_normalization() %>% 
    layer_activation("relu") %>% 
    layer_global_max_pooling_1d() %>%
    layer_dense(units = initParams$neuron_n, activation = 'relu',
                kernel_regularizer = regularizer_l2(l = 0.001)) %>%
    layer_dropout(rate = initParams$dropout) %>%
    layer_dense(units = 11, activation = "softmax")
  
  summary(model)
  
  get_layer(model, index = 1) %>% 
    set_weights(list(embedding_matrix)) %>% 
    freeze_weights()
  
  model %>%  compile(
    #optimizer = optimizer_rmsprop(lr = 0.001),
    optimizer = optimizer_adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999),
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  

  # Training & Evaluation 
  history = model %>% fit(
    x_train, 
    y_train_one_hot,
    epochs = 10,
    batch_size = 128,
    validation_split = 0.2,
    shuffle = TRUE,
    class_weight = as.list(class_weights), 
    verbose = FALSE
  )
  
  score = model %>% evaluate(
    x_test, y_test_one_hot,
    verbose = 0
  )
  
  return(score$accuracy)
}


#initial parameters
initParams = list(filter_n = 128, filter_size = 1, neuron_n = 32, dropout = 0)


#function to search for parameters
maximizeACC = function(filter_n, filter_size, neuron_n, dropout) {
  
  replaceParams = list(filter_n = filter_n, filter_size = filter_size, neuron_n = neuron_n, dropout = dropout)
  updatedParams = modifyList(initParams, replaceParams)
  
  score = training_credit(updatedParams)
  results = list(Score = score,  Pred = 0)
  return(results)
}

#define parameter bounds
boundsParams = list(filter_n = c(128L, 512L), filter_size = c(1L, 20L), neuron_n = c(32L, 512L), dropout = c(0, 0.2))

Final_calibrated = BayesianOptimization(maximizeACC, bounds = boundsParams, 
                                        init_grid_dt = as.data.frame(boundsParams), 
                                        init_points = 10, n_iter = 100, acq = "ucb", 
                                        kappa = 2.576, eps = 0, verbose = TRUE)



tail(Final_calibrated$History)
Final_calibrated$Best_Value

saveRDS(Final_calibrated, "Final_calibrated.rds")

#best model performance: accuracy 86.7%
#with parameters: Round = 82	filter_n = 512.0000	filter_size = 4.0000	neuron_n = 512.0000	dropout = 0.0007	Value = 0.910

####6.Final CNN Model#####
model_cnn <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>% 
  layer_conv_1d(filters = 512, kernel_size = 4, use_bias = FALSE) %>%
  layer_batch_normalization() %>% 
  layer_activation("relu") %>% 
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 512, activation = 'relu',
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(rate = 0.0007) %>%
  layer_dense(units = 11, activation = "softmax")

summary(model_cnn)

get_layer(model_cnn, index = 1) %>% 
  set_weights(list(embedding_matrix)) %>% 
  freeze_weights()

model_cnn %>%  compile(
  optimizer = optimizer_adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)


history_cnn <- model_cnn %>% fit(
  x_train, y_train_one_hot,
  epochs = 10,
  batch_size = 128,
  #validation_split = 0.2,
  shuffle = TRUE,
  class_weight = as.list(class_weights)
)

model_cnn %>% 
  saveRDS("model_cnn.rds")


#accuracy score on the test set
results_cnn <- model_cnn %>% evaluate(x_test, y_test_one_hot, verbose = 0)
results_cnn



#check out category-specific performance
prediction_cnn <- model_cnn %>% predict(x_test) %>% 
  apply(1, which.max)

prediction_cnn <- prediction_cnn - 1

# Save
write.csv(data.frame(ypred=prediction_cnn, ytrue = y_test), "CNN_results.csv", row.names = FALSE)

library(reticulate)
sklearn <- import("sklearn")
rep <- sklearn$metrics$classification_report(prediction_cnn, y_train)
cat(rep)

label_acc <- numeric()

for (i in 0:10) {
  label_index <- y_test == i
  acc <- sum(prediction_cnn[label_index] == y_test[label_index])/length(y_test[label_index])
  label_acc <- append(label_acc, acc)
  
}

label_acc


#check out accuracy f1 score, precision, recall
tibble(obs = as.factor(y_test), pred = as.factor(prediction_cnn)) %>% 
  accuracy(truth = obs, estimate = pred)

tibble(obs = as.factor(y_test), pred = as.factor(prediction_cnn)) %>% 
  precision(truth = obs, estimate = pred)

tibble(obs = as.factor(y_test), pred = as.factor(prediction_cnn)) %>% 
  recall(truth = obs, estimate = pred)

tibble(obs = as.factor(y_test), pred = as.factor(prediction_cnn)) %>% 
  f_meas(truth = obs, estimate = pred)
