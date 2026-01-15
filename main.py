from matplotlib.lines import lineStyles
from scipy import LowLevelCallable
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("/content/stock_data.csv", parse_dates=True , index_col='Date')

df.drop(columns='Unnamed: 0' , inplace=True)

df.head()

sns.set(style="whitegrid")
plt.figure(figsize=(12,6))
sns.lineplot(data=df ,  x='Date' , y='High' , label='High  Price ', color='blue')

plt.title('High and Low Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

df_resampled = df.resample('ME').mean(numeric_only=True)
sns.set(style="whitegrid")
plt.figure(figsize=(12,6))
sns.lineplot(data=df_resampled ,  x=df_resampled.index , y='High', label='Month Wise Average High Price' , color='blue')
plt.xlabel('Date (Monthly)')
plt.ylabel('Price')
plt.title('Monthly Average High Price')
plt.show()

if 'Data' not in df.columns:
    print("'Data' is  already the index or not present in the DataFrame.")
else:
  df.set_index('Data', inplace=True)


plt.figure(figsize=(12,6))
plot_acf(df['High'], lags=40)
plt.title('Autocorrelation Function (ACF) for High Price')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

result = adfuller(df['High'])
print('ADF Statistic: ', result[0])
print('p-value: ', result[1])
print('Critical Values:', result[4])


df['High_diff'] = df['High'].diff()
plt.figure(figsize=(12,6))
plt.plot(df['High']  , label='Original High Price' , color='blue')
plt.plot(df['High_diff'] , label='Differenced High Price' , linestyle='--', color='red')
plt.legend()
plt.title( ' Original  vs Differenced High Price')
plt.show()


window_size = 120
df['high_smoothed'] = df['High'].rolling(window=window_size).mean()

plt.figure(figsize=(12, 6))

plt.plot(df['High'], label='Original High', color='blue')
plt.plot(df['high_smoothed'], label=f'Moving Average (Window={window_size})', linestyle='--', color='orange')

plt.xlabel('Date')
plt.ylabel('High')
plt.title('Original vs Moving Average')
plt.legend()
plt.show()


df_combined = pd.concat([df['High'], df['High_diff']], axis=1)

print(df_combined.head())


df.dropna(subset=['High_diff'], inplace=True)
df['High_diff'].head()





result = adfuller(df['High_diff'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])
