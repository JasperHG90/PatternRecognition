####script to convert the pre-processed pickle data to csv in R and save it as "WikiVital.csv"

library(reticulate)
library(tidyverse)

pd <- import("pandas")

#import word embedding
pickle_data_embedding <- pd$read_pickle("./embedding_matrix.pickle")

#save word embedding as rds
pickle_data_embedding %>% 
  write_rds("./Qixiang/embedding_matrix.rds")


#import training/test data and save them as separate rds files
pickle_data_input <- pd$read_pickle("./vectorized_input_data.pickle")

pickle_data_input$train_x %>% 
  write_rds("./Qixiang/train_x.rds")

pickle_data_input$test_x %>% 
  write_rds("./Qixiang/test_x.rds")

pickle_data_input$train_y %>% 
  write_rds("./Qixiang/train_y.rds")

pickle_data_input$test_y %>% 
  write_rds("./Qixiang/test_y.rds")