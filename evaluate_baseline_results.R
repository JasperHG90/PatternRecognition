## Look at the results of the model
m <- "results/baselineNN_trials.csv"
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
         test_acc = io$test_accuracy,
         test_f1 = io$test_f1) %>%
  mutate(use_batch_norm = ifelse(use_batch_norm == "True", TRUE, FALSE),
         quantilegrp = cut(loss, quantile(loss)),
         optimizer_adam = ifelse(optimizer == "Adam", TRUE, FALSE),
         weighted_loss = ifelse(weighted_loss == "True", TRUE, FALSE)) %>%
  select(-optimizer)

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

# Across optimizers
ggplot(hparams_processed %>%
         mutate(iteration = 1:n()), aes(x=iteration, y=learning_rate, color=optimizer_adam)) +
         geom_point()

# Learning rate versus dropout
p2 <- ggplot(hparams_processed, aes(x = log(learning_rate), y = dropout)) +
  geom_point() +
  geom_vline(aes(xintercept = hparams_processed %>%
                   filter(loss == min(loss)) %>%
                   select(learning_rate) %>%
                   pull() %>%
                   log(.)),
             color = "blue", linetype = "dashed", 
             size = 1.2) +
  geom_hline(aes(yintercept = hparams_processed %>%
                   filter(loss == min(loss)) %>%
                   select(dropout) %>%
                   pull() ),
             color = "green", linetype = "dashed", 
             size = 1.2)
ggExtra::ggMarginal(p2, type = "histogram")

# Number of hidden units
ggplot(hparams_processed %>% group_by(hidden_units) %>%
         tally(), aes(x= hidden_units, y = n)) +
    geom_bar(stat = "identity")
# Model clearly favours larger architecture

ggplot(hparams_processed %>% group_by(filter_size_1) %>%
         tally(), aes(x= filter_size_1, y = n)) +
  geom_bar(stat = "identity")

ggplot(hparams_processed, aes(x=learning_rate, y = loss, color=optimizer_adam)) +
  geom_point() +
  geom_smooth()
# Anything beyond 0.002 is a waste of time basically.

ggplot(hparams_processed, aes(x=dropout, y = loss, shape = optimizer_adam)) +
  geom_point() +
  geom_smooth()
# Less clear. Trend seems to be: more dropout: lower test set performance

#library(plot3D)
#scatter3D(hparams_processed$learning_rate,
#          hparams_processed$dropout,
#          hparams_processed$loss)

library(plotly)
plot_ly(hparams_processed, x =~log(learning_rate), y=~dropout, z=~loss, color = ~quantilegrp,
        symbol = ~use_batch_norm, symbols = c("circle", "x")) %>%
  add_markers()
