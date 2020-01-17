#################################Start Here########################################
library(keras)
library(tidyverse)
library(rBayesianOptimization)


####1.Read Data####
WikiVital <- read_csv("./data/WikiVital.csv")
   
WikiVital <- WikiVital %>%
  group_by(paper) %>%
  mutate(content = paste(sentence, collapse  = "")) %>%
  ungroup() %>%
  select(-sentence) %>%
  distinct() %>%
  mutate(label = group_indices(.,category)) %>%
  mutate(label = label - 1)

text <- pull(WikiVital, content)
labels <- pull(WikiVital, label)
table(labels)



####2.Tokenization####
max_words <- 15000
maxlen <- 150
tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(text)

sequences <- texts_to_sequences(tokenizer, text)

word_index <- tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")

data <- pad_sequences(sequences, maxlen = maxlen)

labels <- as.array(labels)

cat("Shape of data tensor:", dim(data), "\n")
cat('Shape of label tensor:', dim(labels), "\n")

class_weights <- prod(table(labels))/table(labels)/(10^29)

####3.Split Data####
#take a stratified sample as training set
set.seed(856)

index_train <- WikiVital %>%
  rowid_to_column("id") %>%
  group_by(category) %>% 
  sample_frac(0.95) %>%
  ungroup() %>% 
  sample_frac(1L) %>% 
  pull(id)

x_train <- data[index_train, ]
y_train <- labels[index_train]

y_train_one_hot <- to_categorical(y_train)

dim(y_train_one_hot)
y_train_one_hot[1,]

#test set
x_test <- data[-index_train, ]
y_test <- labels[-index_train]

y_test_one_hot <- to_categorical(y_test)
table(y_test)



####4.Wiki word embeddings####
#load pre-trained embeddings
lines <- readLines('./embeddings/wiki-news-300d-1M.vec')
embeddings_index <- new.env(hash = TRUE, parent = emptyenv())
for (i in 1:length(lines)) {
  line <- lines[[i]]
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
}
cat("Found", length(embeddings_index), "word vectors.\n")

embeddings_index[["word"]]


#prepare word-embeddings matrix
embedding_dim <- 300
embedding_matrix <- array(0, c(max_words, embedding_dim))
for (word in names(word_index)) {
  index <- word_index[[word]]
  if (index < max_words) {
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector))
      embedding_matrix[index+1,] <- embedding_vector
  }
}
dim(embedding_matrix)


####5.Single CNN#####
model_cnn <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>% 
  layer_conv_1d(filters = 512, kernel_size = 10, use_bias = FALSE) %>%
  layer_batch_normalization() %>% 
  layer_activation("relu") %>% 
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 512, activation = 'relu',
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(rate = .1) %>%
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
  validation_split = 0.2,
  shuffle = TRUE,
  class_weight = as.list(class_weights)
)

results_cnn <- model_cnn %>% evaluate(x_test, y_test_one_hot, verbose = 0)
results_cnn

#check out category-specific performance
prediction_cnn <- model_cnn %>% predict(x_test) %>% 
  apply(1, which.max)

prediction_cnn <- prediction_cnn - 1

label_acc <- numeric()

for (i in 0:10) {
  label_index <- y_test == i
  acc <- sum(prediction_cnn[label_index] == y_test[label_index])/length(y_test[label_index])
  label_acc <- append(label_acc, acc)
  
}

label_acc

####5.Bayesian Optimization for Parameter Search####
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
boundsParams = list(filter_n = c(128L, 512L), filter_size = c(1L, 10L), neuron_n = c(32L, 512L), dropout = c(0, 0.1))

Final_calibrated = BayesianOptimization(maximizeACC, bounds = boundsParams, 
                                        init_grid_dt = as.data.frame(boundsParams), 
                                        init_points = 10, n_iter = 30, acq = "ucb", 
                                        kappa = 2.576, eps = 0, verbose = TRUE)



tail(Final_calibrated$History)

#best model performance: accuracy 86.7%
#with parameters: filter_n = 512, filter_size =10, neuron_n = 512, dropout = 0.1
