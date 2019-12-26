# PatternRecognition
Project files for pattern recognition group assignment

## Files

Currently contains the following files:

1. `data/WikiEssentials_L4.7z`: output file of the WikiVitalArticles program. Each document is included in its entirety (but split by paragraph).  
1. `preprocess_utils.py`: preprocessing functions for Wiki data.
2. `model_utils.py`: various utility functions used for modeling (e.g. loading embeddings).
3. `1_preprocess_raw_data.py`: preprocessing of raw input data. Currently shortens each article to first 8 sentences. 
4. `2_baseline_model.py`: tokenization, vectorization of input data and baseline model (1-layer NN with softmax classifier). 

## Setup

1. Download and install [Anaconda Python 3](https://www.anaconda.com/distribution/)
2. Download latest version of [Rstudio](https://rstudio.com/products/rstudio/download/). Need this to run python scripts in Rstudio.
3. In a terminal, go to this repository's folder and set up the Conda environment

```shell
conda env create -f environment.yml
```

4. Install the additional requirements

```shell
conda activate VitalWikiClassifier
pip install -r requirements.txt
```

5. In R, install the `reticulate` library:

```r
install.packages("reticulate")
```

6. Check the `.Rprofile` file to ensure that R knows where to find your anaconda distribution.

