![Zillow-Kaggle](https://github.com/ddhaval04/Zillow-Kaggle-Competition/raw/master/images/zillow.jpeg)


# Overview

This is the code for my submission in the Kaggle Zillow's Home Value Prediction competition. The solution at the time of the competition was ranked in the *Top 20%*.

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

# Some interesting insights from exploratory analysis:

To explore more analysis, please refer the `script/Zillow_Exploration.ipynb` notebook.

## 1)

![Zillow-Kaggle](https://github.com/ddhaval04/Zillow-Kaggle-Competition/raw/master/images/log-error-distribution.png)

From the above graph we can tell that if the logerror is positive, it means that Zillow is overestimating the Saleprice, and if the logerror is negative, it means that Zillow is underestimating the Salesprice.

## 2)

![Zillow-Kaggle](https://github.com/ddhaval04/Zillow-Kaggle-Competition/raw/master/images/houses-build-year.png)

There has been a huge rise in the number of houses build in the recent year.

## 3)

![Zillow-Kaggle](https://github.com/ddhaval04/Zillow-Kaggle-Competition/raw/master/images/log-error-over-the-years.png)

As per the data, Zillow has been fairly recently facing a problem of accurately predicting the prices of the houses. This can be seen from the consistent increase in the `logerror_mean` over the months.

# Credits:

- Kaggle community