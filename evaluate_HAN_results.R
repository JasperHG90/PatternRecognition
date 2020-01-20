## Look at the results of the model
m <- "results/HAN_trials.csv"
io <- read.csv(m, stringsAsFactors = FALSE)
# Get json
hparams <- io$params
hparams[1]
# Turn json into structure
outpar <- function(x) {
  # Prep string
  out <- gsub("\\{|\\}|'", "", x)
  out <- strsplit(out, ",")[[1]]
  out <- lapply(out, function(y) strsplit(y, ":"))
  out <- unlist(out, recursive = FALSE)
  # Reconstruct output
  out_new <- list()
  for(i in seq_along(out)) {
    po <- out[[i]]
    out_new[[trimws(po[1])]] <- tryCatch({
      as.numeric(trimws(po[2]))
    }, warning = function(e) {
      trimws(po[2])
    }, error = function(e) {
      trimws(po[2])
    }) 
  }
  # Return
  return(out_new)
}
library(purrr)
library(dplyr)
hparams_processed <- map_df(hparams, outpar) %>%
  mutate(loss = io$val_loss,
         val_acc = io$val_accuracy,
         val_loss = io$val_loss,
         val_f1 = io$val_f1) %>%
  mutate(quantilegrp = cut(val_loss, quantile(val_loss)),
         weighted_loss = ifelse(use_class_weights == "True", TRUE, FALSE)) %>%
  select(-use_class_weights)

# Plot some density plots
library(ggplot2)
library(ggExtra)

# Plot params across trends
library(tidyr)
i <- hparams_processed %>%
  mutate(iteration = 1:n()) %>%
  select(-quantilegrp) %>%
  gather(variable, value, -iteration) %>%
  ggplot(., aes(x=iteration, y =value, color = variable)) +
  geom_point() +
  geom_smooth(se = FALSE, color = "grey") +
  theme_bw() +
  facet_wrap(.~ variable, nrow = 3, scales = "free_y")
i

# Learning rate
ggplot(hparams_processed, aes(x=(learning_rate))) +
  geom_density() +
  geom_vline(aes(xintercept = hparams_processed %>%
                   filter(loss == min(loss)) %>%
                   select(learning_rate) %>%
                   pull()),
             color = "blue", linetype = "dashed", 
             size = 1.2)
# Mean learning rate
exp(mean(log(hparams_processed$learning_rate)))

# Number of hidden units
ggplot(hparams_processed %>% group_by(hidden_size) %>%
         tally(), aes(x= hidden_size, y = n)) +
  geom_bar(stat = "identity")
# Model clearly favours larger architecture

ggplot(hparams_processed %>% group_by(sent_length) %>%
         tally(), aes(x=sent_length, y=n)) +
  geom_bar(stat = "identity")

ggplot(hparams_processed, aes(x=learning_rate, y = loss)) +
  geom_point() +
  geom_smooth()


