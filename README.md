![Zillow-Kaggle](https://github.com/ddhaval04/Zillow-Kaggle-Competition/raw/master/images/zillow.jpeg)


# Overview

This is the code for my submission in the Kaggle Zillow's Home Value Prediction competition. The solution at the time of the competition was ranked in the **Top 20%**.

# Challenge

To improve the algorithm that changed the world of real estate!

# Dependencies (pip install)

```
numpy
pandas
matplotlib
seaborn
sklearn
keras
```

# Usage

- To use the pre-trained model:
```
To run the XGBoost model: (submitted to the competition)

python -pt xg script.py
```

- To train your model from scratch:
```
To train the XGBoost model: (submitted to the competition)

python xg script.py

To train the Neural Network :

python nn script.py
```

### To have a look at the data exploration, please refer the `Scripts/Zillow_Exploration.ipynb` notebook.

# Results:

Submissions were evaluated on Mean Absolute Error between the predicted log-error and the actual log-error. The MAE obtained by my script is `0.0658257` and was ranked in **Top 20%** on the public and private leaderboard.

# Credits:

- Kaggle community