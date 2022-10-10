# About

The Project IsoCrypto was inspired by statistical tools applied in the data mining of astrophysics data. The main references were theoretical curves adjusted in open clusters intending to understand the fundamental parameters that make its stars – these curves are called isochrones.

Given such use in astrophysics, this project aims to amplify the applications of those curves in econophysics. The goal is to analyze temporal series of financial data in real-time, verifying if those stochastic events are similar to others that occurred in the past (a search for patterns), and through that, to predict the price of assets to make decisions.

To execute the temporal analysis, the variation of bitcoin prices was used, due to the cryptocurrency market working 24h a day – therefore there’s no interruption in the data flux. The Project, however, can also be applied to stocks, for example, with the appropriate adjustments.

## **Methods**

This project used cross entropy as its methodology for its two distributions: the first one are the data that make the dataset, and the second one is the real-time data that are denoted as “target” in the software.

The target’s temporal curve is compared to over curves of the past, thus returning the curve of the closest dataset that explains the present one. Assuming that such pattern will repeat itself, the prediction can be made for a short time period.

To consider only the best-adjusted curves, a filter is applied to return only the best results of the adjustment coefficients (R²) to assist in decision-making. In addition to that, through the dashboard that was developed, it’s possible to see the adjustments in real time.

![Projeto sem título](https://user-images.githubusercontent.com/70342250/194788284-2a4a7232-ac86-40ac-9b1a-d25ba3853b53.gif)

## Status

Under development and improvement, mainly the backtest tool and the creation of theoretical curves from the dataset.

## Run

To perform predict adjustments, initially, a dataset must be created, this dataset must be created only once and updated using the "update_dataset".

To perform forecast adjustments, initially a dataset must be created, this dataset must be created only once and updated using the "update_dataset" file.

Two options are available for tracking forecasts:

- The "predict" file when executed displays its results through matplotlib. In this option, a .png file is saved when the R² is greater than 0.80.
- The file "predict_real_time" when executed displays the adjustments in real time through an interface built in Streamlit.

## Requirements

- Streamlit
- Ployly
- Numpy
- Pandas
- Python-binance
- Scikit-learn
- Matplotlib
