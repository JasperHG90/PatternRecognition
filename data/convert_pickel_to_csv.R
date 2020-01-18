####script to convert the pre-processed pickle data to csv in R and save it as "WikiVital.csv"

library(reticulate)
library(tidyverse)
#import data
pd <- import("pandas")
pickle_data <- pd$read_pickle("WikiEssentials_L4_P3_preprocessed.pickle")

#inspect data
pickle_data$History$DOC669
pickle_data[[1]]

dim(pd$DataFrame(pickle_data))

as.tibble(pd$DataFrame(pickle_data)) %>%
  gather(category, paper) %>%
  .[1,2] %>%
  dim()

#transform data from pickle to data frame
dfs <- lapply(pickle_data, data.frame, stringsAsFactors = FALSE)

good_data <- bind_rows(dfs, .id = "category") %>%
  gather(paper, sentence, -category) %>%
  as_tibble() %>%
  drop_na()

#save data
good_data %>%
  write_csv("WikiVital.csv")