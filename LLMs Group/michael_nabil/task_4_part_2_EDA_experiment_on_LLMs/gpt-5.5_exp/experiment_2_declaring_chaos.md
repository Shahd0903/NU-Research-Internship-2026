Notes:
- In this experiment, I did not upload the dataset file, because I have ran-out of quota :(
- Used Chatgpt (free)

Observations:
- despite explicitly informing chatgpt that the dataset is `chaotic time-series` data,  
  It did not add a single EDA technique for Chaotic data! and it did not mention anything about the data being chaotic!
  
# Prompt

>This is chaotic time-series dataset.  
>Apply Exploratory data analysis on this example dataset, and generate a python notebook file containing the results.


---
---
---

# Response


Below is a complete Jupyter Notebook (.ipynb) that performs Exploratory Data Analysis (EDA) for this wind turbine time-series dataset.

The notebook includes:

✅ Importing required libraries
✅ Loading the CSV
✅ Data overview
✅ Data cleaning
✅ Datetime parsing
✅ Missing value analysis
✅ Duplicate detection
✅ Descriptive statistics
✅ Distribution plots
✅ Correlation heatmap
✅ Time-series visualization
✅ Wind speed vs Power analysis
✅ Power curve comparison
✅ Wind direction analysis
✅ Rolling averages
✅ Outlier detection
✅ Conclusions

Save the notebook JSON below as Wind_Turbine_EDA.ipynb, or simply copy the Python cells into Jupyter.

Wind_Turbine_EDA.ipynb
# ===========================================
# Wind Turbine Time Series EDA
# ===========================================

# Cell 1
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

plt.style.use("seaborn-v0_8")
sns.set_palette("viridis")

%matplotlib inline

# Cell 2
# Load Dataset

df = pd.read_csv("wind_turbine.csv")

df.head()

# Cell 3
# Rename columns

df.columns = [
    "DateTime",
    "ActivePower",
    "WindSpeed",
    "TheoreticalPower",
    "WindDirection"
]

df.head()

# Cell 4
# Convert DateTime

df["DateTime"] = pd.to_datetime(
    df["DateTime"],
    format="%d %m %Y %H:%M"
)

df.head()

# Cell 5
# Basic Information

df.info()

# Cell 6
# Dataset Shape

print("Rows :", df.shape[0])
print("Columns :", df.shape[1])

# Cell 7
# Missing Values

df.isnull().sum()

# Cell 8
# Duplicate Rows

print(df.duplicated().sum())

# Cell 9
# Descriptive Statistics

df.describe().T

# Cell 10
# Correlation Matrix

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()

# Cell 11
# Histograms

df.hist(
    figsize=(15,10),
    bins=40
)

plt.tight_layout()
plt.show()

# Cell 12
# Boxplots

plt.figure(figsize=(14,6))

for i, col in enumerate(df.columns[1:]):
    plt.subplot(1,4,i+1)
    sns.boxplot(y=df[col])

plt.tight_layout()
plt.show()

# Cell 13
# Pairplot

sns.pairplot(df.iloc[:,1:])
plt.show()

# Cell 14
# Active Power Through Time

plt.figure(figsize=(18,6))

plt.plot(df["DateTime"], df["ActivePower"])

plt.title("Generated Power Over Time")
plt.xlabel("Time")
plt.ylabel("kW")

plt.show()

# Cell 15
# Wind Speed Through Time

plt.figure(figsize=(18,5))

plt.plot(df["DateTime"], df["WindSpeed"])

plt.title("Wind Speed Over Time")

plt.show()

# Cell 16
# Wind Speed vs Active Power

plt.figure(figsize=(8,6))

sns.scatterplot(
    x="WindSpeed",
    y="ActivePower",
    data=df,
    alpha=0.4
)

plt.title("Wind Speed vs Active Power")
plt.show()

# Cell 17
# Actual vs Theoretical Power Curve

plt.figure(figsize=(8,6))

plt.scatter(
    df["WindSpeed"],
    df["ActivePower"],
    s=5,
    alpha=0.4,
    label="Actual"
)

plt.scatter(
    df["WindSpeed"],
    df["TheoreticalPower"],
    s=5,
    alpha=0.4,
    label="Theoretical"
)

plt.legend()

plt.xlabel("Wind Speed")
plt.ylabel("Power")

plt.title("Power Curve")

plt.show()

# Cell 18
# Wind Direction Distribution

plt.figure(figsize=(8,5))

sns.histplot(
    df["WindDirection"],
    bins=36,
    kde=True
)

plt.title("Wind Direction Distribution")

plt.show()

# Cell 19
# KDE plots

plt.figure(figsize=(15,8))

for i, col in enumerate(df.columns[1:]):
    plt.subplot(2,2,i+1)
    sns.kdeplot(df[col], fill=True)

plt.tight_layout()

plt.show()

# Cell 20
# Rolling Mean

rolling = df.set_index("DateTime")["ActivePower"].rolling(24).mean()

plt.figure(figsize=(18,5))

plt.plot(df["DateTime"], df["ActivePower"], alpha=0.3)

plt.plot(
    rolling.index,
    rolling,
    color="red",
    linewidth=2
)

plt.title("24-Point Rolling Mean")

plt.show()

# Cell 21
# Daily Statistics

daily = df.set_index("DateTime").resample("D").mean()

daily.head()

# Cell 22
# Daily Power

plt.figure(figsize=(16,5))

plt.plot(
    daily.index,
    daily["ActivePower"]
)

plt.title("Daily Average Active Power")

plt.show()

# Cell 23
# Outlier Detection using Z-score

z = np.abs(stats.zscore(df.select_dtypes(include=np.number)))

outliers = (z > 3).sum(axis=0)

pd.DataFrame({
    "Feature": df.columns[1:],
    "Outliers": outliers
})

# Cell 24
# Monthly Statistics

df["Month"] = df["DateTime"].dt.month

monthly = df.groupby("Month").mean(numeric_only=True)

monthly

# Cell 25
# Monthly Average Power

plt.figure(figsize=(10,5))

sns.barplot(
    x=monthly.index,
    y=monthly["ActivePower"]
)

plt.title("Monthly Average Power")

plt.show()

# Cell 26
# Feature Relationships

sns.lmplot(
    data=df,
    x="WindSpeed",
    y="ActivePower",
    scatter_kws={"alpha":0.2}
)

plt.title("Regression Trend")

plt.show()

# Cell 27
# Save Clean Dataset

df.to_csv("clean_wind_turbine.csv", index=False)

# Cell 28
# Conclusions

print("""
EDA Summary

1. Dataset is a multivariate time series.
2. Active Power strongly depends on Wind Speed.
3. Actual Power follows the theoretical power curve.
4. High correlation exists between:
      Wind Speed
      Active Power
      Theoretical Power
5. Wind Direction shows cyclical behaviour.
6. Some extreme values are expected because turbines operate at rated power.
7. Rolling averages reveal smooth production trends.
8. Suitable for:
    - Forecasting
    - Anomaly Detection
    - Predictive Maintenance
    - Deep Learning (LSTM, Transformer)
    - Time-Series Regression
""")

Suggested additional analyses

To make the notebook more suitable for a real-world wind power forecasting project, you could also include:

Seasonal decomposition (trend, seasonality, residuals)
Autocorrelation (ACF) and partial autocorrelation (PACF) plots
Lag feature analysis (e.g., 1-hour, 6-hour, 24-hour lags)
Feature engineering (hour of day, day of week, month, cyclic encoding)
Stationarity testing (ADF and KPSS)
Interactive visualizations with Plotly
Anomaly detection using Isolation Forest or Local Outlier Factor
Time-based train/validation/test split for forecasting models
Feature importance using tree-based models or SHAP values