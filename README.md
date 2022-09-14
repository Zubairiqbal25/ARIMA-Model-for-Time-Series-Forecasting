# ARIMA-Model-for-Time-Series-Forecasting

<p>In this notebook, I will discuss ARIMA Model for time series forecasting. ARIMA model is used to forecast a time series using the series past values. In this notebokk, we build an optimal ARIMA model and extend it to Seasonal ARIMA (SARIMA) and SARIMAX models. We will also see how to build autoarima models in python.</p>

<h2>import Following libraries</h2>
 
```
!pip install pmdarima==1.8.5
!pip install statsmodels==0.12.2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

from statsmodels.tsa.stattools import adfuller
from numpy import log

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA

import pmdarima as pm

```

<h2>Upload file</h2>

```
path = '/content/usertrack1.csv'
df = pd.read_csv(path)
df.head()
```
<h3>Output</h3>
![image](https://user-images.githubusercontent.com/28058334/190084599-e856815f-ce48-4b16-bed4-02ac0b23695b.png)

```
result = adfuller(df.Polarity.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
```

<h2>Find Difference and Orignal Data correlation</h2>

```
# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.Polarity); axes[0, 0].set_title('Original Series')
plot_acf(df.Polarity, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df.Polarity.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.Polarity.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df.Polarity.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.Polarity.diff().diff().dropna(), ax=axes[2, 1])

plt.show()
```
<h3>Output</h3>
![image](https://user-images.githubusercontent.com/28058334/190085207-228225e0-5c95-4bdf-bf8f-438d610dd3b4.png)

<h2>PACF plot of 1st differenced series</h2>

```
# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.Polarity.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.Polarity.diff().dropna(), ax=axes[1])

plt.show()
```

<h3>Output</h3>
![image](https://user-images.githubusercontent.com/28058334/190086669-01424800-c45e-4e0b-a926-ece202a71b32.png)

```

plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.Polarity.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(df.Polarity.diff().dropna(), ax=axes[1])

plt.show()

```
<h3>Output</h3>
![image](https://user-images.githubusercontent.com/28058334/190086875-d25e6177-dbf5-4dd2-a1ba-239856105c8b.png)


# 1,1,2 ARIMA Model
model = ARIMA(df.Polarity, order=(1,1,2))
model_fit = model.fit()
print(model_fit.summary())

<h3>Output</h3>
![image](https://user-images.githubusercontent.com/28058334/190087182-0090b1ad-7814-4bfd-8069-f7e4d2f24f9f.png)


```
# 1,1,1 ARIMA Model
model = ARIMA(df.Polarity, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())
```

```
# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
```

<h3>Output</h3>
![image](https://user-images.githubusercontent.com/28058334/190087428-9714e5eb-71fa-4ac7-90c3-fbcc9b078076.png)



```
# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()
```

<h3>Output</h3>
![image](https://user-images.githubusercontent.com/28058334/190087498-38e8cc7e-4c28-4682-a6be-9c37caced407.png)

```

# Create Training and Test
train = df.Polarity[:85]
test = df.Polarity[85:]
```


```
# Build Model
# model = ARIMA(train, order=(3,2,1))  

model = ARIMA(train, order=(1, 1, 1))  
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(15, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()
```
<h3>Output</h3>
![image](https://user-images.githubusercontent.com/28058334/190087749-98aebfeb-91f1-4fe5-bd9e-0f86fa5851c6.png)

```
# Build Model
model = ARIMA(train, order=(3, 2, 1))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

# Forecast
fc, se, conf = fitted.forecast(15, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()
```
<h3>Output</h3>
![image](https://user-images.githubusercontent.com/28058334/190087828-b1438aa8-acd5-4d7c-85b6-a2e20dd52e00.png)

```
# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)
```

```
model = pm.auto_arima(df.Polarity, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())
```

```
model.plot_diagnostics(figsize=(10,8))
plt.show()
```
<h3>Output</h3>
![image](https://user-images.githubusercontent.com/28058334/190088066-fa1e587b-8227-4cff-85f0-3db7b901ef0e.png)

<h2>Forecast</h2>
```
n_periods = 24
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df.Polarity), len(df.Polarity)+n_periods)
```
<h2> make series for plotting purpose</h2>
```
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)
```
<h2>Plot</h2>
```
plt.plot(df.Polarity)
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of Usage")
plt.show()
```
<h3>Output</h3>
![image](https://user-images.githubusercontent.com/28058334/190088142-db32a007-7573-48a0-be08-9a8fc146e4db.png)
