# PatternRecognition

Project files for pattern recognition group assignment. Our project is about the classification of Wikipedia articles in one of the 11 top-level categories of the [Vital Articles Wikipedia list, level 4](https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/4).

The vital articles list was downloaded using scrapy. The code can be found in the [WikiVitalArticles](https://github.com/JasperHG90/WikiVitalArticles) repository. The raw data is included in this repository.

Given the different skillsets in our group, we use a mix of R, python, Keras and Pytorch to build our models. However, we make sure that each model uses the same train/test splits.

Our presentation can be found here. Our final paper can be found here.

## Files

Currently contains the following files (not all files are listed here):

1. `data/raw/WikiEssentials_L4.7z`: output file of the WikiVitalArticles program. Each document is included in its entirety (but split by paragraph).  
2. `preprocess_utils.py`: preprocessing functions for Wiki data.
3. `model_utils.py`: various utility functions used for modeling (e.g. loading embeddings).
4. `1_preprocess_raw_data.py`: preprocessing of raw input data. Currently shortens each article to first 8 sentences. 
5. `2_baseline_model.py`: Pytorch implementation of the baseline model (1-layer NN with softmax classifier). 
6. `3_cnn_model.R`: Keras implementation of a 1D convolutional neural network.
7. `4_lstm_model.py`: Pytorch implementation of a Long-Short Term Recurrent Neural Network.
8. `5_han_model.py`: Pytorch implementation of a Hierarchical Attention Network (HAN).
9. `6_statistical_test.R`: Contains R code to perform Stuart-Maxwell test on the classification outcomes.
10. `HAN.py`: Contains the Pytorch module implementation of the HAN.
11. `LSTM.py`: Contains the Pytorch module implementation of the LSTM.

It contains the following folders:

1. `data`: Contains raw and pre-processed data used by the models. To understand te pipeline from raw to preprocessed data, see the `preprocess_utils.py` file.
2. `embeddings`: Folder in which FastText embeddings should be downloaded and unzipped.
3. `img`: Contains images.
4. `model_cnn`: Final model for the convolutional neural network after hyperparameter optimization.
5. `models`: Final Pytorch model weights for the baseline, HAN and LSTM.
6. `predictions`: CSV files containing the predictions and ground-truth labels for each model.
7. `results`: CSV files containing the results of the hyperparameter search we conducted using [Hyperopt](https://github.com/hyperopt/hyperopt).

## Setup

1. Download and install [Anaconda Python 3](https://www.anaconda.com/distribution/)
2. Download latest version of [Rstudio](https://rstudio.com/products/rstudio/download/). Need this to run python scripts in Rstudio.
3. In a terminal, go to this repository's folder and set up the Conda environment

```shell
conda env create -f environment.yml
```

Note that this will install both Python requirements as well as R requirements. We use a separate R library location that is set in the `.Renviron` file.

4. In R, install the following libraries:

```r
install.packages(c("yardstick", "rBayesianOptimization", "DescTools", "ggExtra"))
```

5. Check the `.Rprofile` file to ensure that R knows where to find your anaconda distribution. Check the `.Renviron` file to ensure that the path to the Anaconda environment is set correctly.

## Shiny application

We created a small shiny application that allows you to input a document and visualize the HAN attention predictions and score. Find the repository to this shiny app [here](https://github.com/JasperHG90/shiny_han).

![shiny-app](img/shiny.gif)

