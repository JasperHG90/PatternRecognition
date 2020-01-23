## Perform a statistical test on the predictions of the various models

# Read results
baseline_results <- read.csv("predictions/baseline.csv")
HAN_results <- read.csv("predictions/HAN.csv")
CNN_results <- read.csv("predictions/CNN_results.csv")

# Function to make confusion matrix
MakeConfusionMatrix <- function(ypred, ytrue, num_classes, add_one = TRUE) {
  cmat <- matrix(0L, ncol = num_classes, nrow = num_classes)
  if(add_one) {
    ypred <- ypred + 1
    ytrue <- ytrue + 1
  }
  # Populate matrix
  for(i in seq_along(1:length(ypred))) {
    cmat[ytrue[i], ypred[i]] <- cmat[ytrue[i], ypred[i]] + 1
  }
  # Return
  return(cmat)
}

# Confusion matrices
(BL_vCNN <- MakeConfusionMatrix(baseline_results$yhat, CNN_results$ypred, 11))
(BL_vHAN <- MakeConfusionMatrix(baseline_results$yhat, HAN_results$yhat, 11))
(CNN_vHAN <- MakeConfusionMatrix(CNN_results$ypred, HAN_results$yhat, 11))

# Do statistical test
library(DescTools)
DescTools::StuartMaxwellTest(BL_vCNN)
DescTools::StuartMaxwellTest(BL_vHAN)
DescTools::StuartMaxwellTest(CNN_vHAN)

library(reticulate)
skl <- import("sklearn.metrics")
cat(skl$classification_report(CNN_results$ypred, CNN_results$ytrue))
