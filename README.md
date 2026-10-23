# Drug Dissolution Prediction

A machine learning project for predicting drug dissolution using physicochemical descriptors and experimental formulation parameters.

## Model

The project uses LightGBM regression.

Experimental series are split into training and test sets using GroupShuffleSplit, with Experimental_index as the grouping variable to reduce leakage between measurements from the same series.

## ## Data

The dataset is not included in this repository. Before running the script, place the required Excel dataset in the expected location and ensure that the file path in `model.py` points to it.

## Evaluation

The current model was evaluated using RMSE and R² on a held-out test set. These results represent an initial model evaluation and do not establish generalization to every new drug or formulation.

## Requirements

Python 3.10 or later is recommended.

Install dependencies with:

    pip install -r requirements.txt
