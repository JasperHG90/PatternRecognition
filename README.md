# PatternRecognition

Project files for pattern recognition group assignment. Our project is about the classification of Wikipedia articles in one of the 11 top-level categories of the [Vital Articles Wikipedia list, level 4](https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/4).

The vital articles list was downloaded using scrapy. The code can be found in the [WikiVitalArticles](https://github.com/JasperHG90/WikiVitalArticles) repository. The raw data is included in this repository.

Given the different skillsets in our group, we use a mix of R, python, Keras and Pytorch to build our models. However, we make sure that each model uses the same train/test splits.

Our presentation can be found here. Our final paper can be found here.

## Files

Currently contains the following files (only the most important files are listed here):

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

To run our code, we recommend using [PyCharm](https://www.jetbrains.com/pycharm/) or [VS Code](https://code.visualstudio.com/). The latter can be downloaded in the Anaconda Launcher.

![anaconda-vscode](img/AL.png)

## Setup

1. Download and install [Anaconda Python 3](https://www.anaconda.com/distribution/)
2. Download latest version of [Rstudio](https://rstudio.com/products/rstudio/download/). Need this to run python scripts in Rstudio.
3. If you want to re-run our data pre-processing steps, download the FastText word embeddings from [here](https://fasttext.cc/docs/en/english-vectors.html). You need the `wiki-news-300d-1M.vec.zip` file. Save the file in the `embeddings` folder and unzip it there. If you do not want to rerun the data preprocessing steps, then you don't need the FT embeddings to re-run our models; they are included in the preprocessed data files. 
4. In a terminal, go to this repository's folder and set up the Conda environment

```shell
conda env create -f environment.yml
```

Note that this will install both Python requirements as well as R requirements. We use a separate R library location that is set in the `.Renviron` file.

5. In R, install the following libraries:

```r
install.packages(c("yardstick", "rBayesianOptimization", "DescTools", "ggExtra"))
```

6. Check the `.Rprofile` file to ensure that R knows where to find your anaconda distribution. Check the `.Renviron` file to ensure that the path to the Anaconda environment is set correctly.

## Shiny application

We created a small shiny application that allows you to input a document and visualize the HAN attention predictions and score. Find the repository to this shiny app [here](https://github.com/JasperHG90/shiny_han).

![shiny-app](img/shiny.gif)

## References

The following papers were instrumental for our work.

[1] Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016, June). Hierarchical attention networks for document classification. In Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies (pp. 1480-1489).

[2] Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. Transactions of the Association for Computational Linguistics, 5, 135-146.

[3] Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.

[4] Bergstra, J., Yamins, D., & Cox, D. D. (2013, June). Hyperopt: A python library for optimizing the hyperparameters of machine learning algorithms. In Proceedings of the 12th Python in science conference (pp. 13-20). Citeseer.
