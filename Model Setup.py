import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
import warnings

# Close Warnings
warnings.filterwarnings("ignore")

# Download Tesla Data (Monthly)
tsla_data = yf.download('TSLA', start='2015-01-01', end='2024-12-01', interval='1mo')

# Only Keep Closing Price
tsla_monthly = tsla_data[['Close']].copy()
tsla_monthly.index.name = 'Month'

# Ensure Index is Datetime Type
tsla_monthly.index = pd.to_datetime(tsla_monthly.index)
tsla_monthly = tsla_monthly.asfreq('MS')

# Print Data Check
print("Raw Data Preview:")
print(tsla_monthly.head())

# Split Train and Test Data
train_data = tsla_monthly.loc['2015-01-01':'2023-12-01']
test_data = tsla_monthly.loc['2024-01-01':'2024-12-01']

print("Test Data Preview:")
print(train_data.head())
print("Train Data Preview:")
print(test_data.head())

# Check Missing Values
print("Missing Values:")
print(train_data.isnull().sum())

# Hyperparameter Range Setting (Suggested to Keep Small Range)
p = range(0, 3)
d = range(0, 2)
q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

# Search for Best Parameter Combination
AIC = []
SARIMAX_model = []

print("Starting Parameter Combination...")

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train_data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit(disp=False)
            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except:
            continue

# If No Successful Model, Exit
if not AIC:
    print("No suitable model found, please try reducing model order or check data")
    exit()

# Find Minimum AIC Model
best_aic = min(AIC)
best_index = AIC.index(best_aic)
best_order = SARIMAX_model[best_index][0]
best_seasonal_order = SARIMAX_model[best_index][1]

print(f'\nBest Model: SARIMAX{best_order} x {best_seasonal_order} - AIC: {best_aic:.2f}')

# Fit Best Model
mod = sm.tsa.statespace.SARIMAX(train_data,
                                order=best_order,
                                seasonal_order=best_seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

# Diagnostic Plot
results.plot_diagnostics(figsize=(15, 12))
plt.tight_layout()
plt.show()

# (Optional) Predict Future 12 Months
forecast = results.get_forecast(steps=12)
forecast_ci = forecast.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data, label='Training Data')
plt.plot(test_data.index, test_data, label='Test Data', color='gray')
plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, label='Forecast', color='orange')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='orange', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Tesla Monthly Close Price (USD)')
plt.title('Tesla Stock Price Forecast (SARIMA)')
plt.legend()
plt.grid(True)
plt.show()




