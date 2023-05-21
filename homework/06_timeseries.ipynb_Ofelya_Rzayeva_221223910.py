#Ofelya Rzayeva 221223910
#Exercise_6

#Task 1
#Load the necessary libraries:
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR

# Import data
data = pd.read_csv("02_python_data.csv", index_col=0)
dax_index = data[".GDAXI"]

# Fit ARMA(1,1) model
model_arma_11 = sm.tsa.AutoReg(dax_index, lags=1)
results_arma_11 = model_arma_11.fit()

# Forecast 30 days
forecast_arma_11 = results_arma_11.predict(start=len(dax_index), end=len(dax_index)+29)

# Plot actual data and forecast for ARMA(1,1)
plt.plot(dax_index, label="Actual")
plt.plot(forecast_arma_11, label="ARMA(1,1) Forecast")
plt.title("DAX Index - ARMA(1,1) Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Fit ARMA(5,5) model
model_arma_55 = sm.tsa.AutoReg(dax_index, lags=5)
results_arma_55 = model_arma_55.fit()

# Forecast 30 days
forecast_arma_55 = results_arma_55.predict(start=len(dax_index), end=len(dax_index)+29)

# Plot actual data and forecast for ARMA(5,5)
plt.plot(dax_index, label="Actual")
plt.plot(forecast_arma_55, label="ARMA(5,5) Forecast")
plt.title("DAX Index - ARMA(5,5) Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

#task 2
# Load data
gdaxi_returns = np.log(data[".GDAXI"] / data[".GDAXI"].shift(1))
constituents_returns = np.log(data.iloc[:, 1:] / data.iloc[:, 1:].shift(1))

constituents_returns = constituents_returns.dropna()
gdaxi_returns = gdaxi_returns.dropna()

# Fit the linear regression model for the replication portfolio
model = LinearRegression()
model.fit(constituents_returns, gdaxi_returns)
weights = model.coef_

# Calculate replication portfolio returns
replication_portfolio_returns = np.dot(constituents_returns, weights)

# Perform cointegration test for replication portfolio
score, pvalue, _ = coint(replication_portfolio_returns, gdaxi_returns)
if pvalue < 0.05:
    print("There is evidence of cointegration between the replication portfolio and the DAX index.")
else:
    print("There is no evidence of cointegration between the replication portfolio and the DAX index.")

# Fit the linear regression model for the smart-beta portfolio
top_3_constituents = constituents_returns.corrwith(gdaxi_returns).abs().nlargest(3).index
subset_constituents_returns = constituents_returns[top_3_constituents]
model_subset = LinearRegression()
model_subset.fit(subset_constituents_returns, gdaxi_returns)
weights_subset = model_subset.coef_

# Calculate smart-beta portfolio returns
smart_beta_portfolio_returns = np.dot(subset_constituents_returns, weights_subset)

# Perform cointegration test for smart-beta portfolio
score_subset, pvalue_subset, _ = coint(smart_beta_portfolio_returns, gdaxi_returns)
if pvalue_subset < 0.05:
    print("There is evidence of cointegration between the smart-beta portfolio and the DAX index.")
else:
    print("There is no evidence of cointegration between the smart-beta portfolio and the DAX index.")

#Task 3
# Load data
returns = np.log(data / data.shift(1)).dropna()

# Perform pairwise cointegration tests
num_assets = returns.shape[1]
cointegration_matrix = np.zeros((num_assets, num_assets))

for i in range(num_assets):
    for j in range(i+1, num_assets):
        asset1 = returns.iloc[:, i]
        asset2 = returns.iloc[:, j]
        result = adfuller(asset1 - asset2)
        pvalue = result[1]
        
        if pvalue < 0.05:
            cointegration_matrix[i, j] = 1

# Print the cointegration matrix
columns = returns.columns
cointegration_df = pd.DataFrame(cointegration_matrix, index=columns, columns=columns)
print("Cointegration Matrix:")
print(cointegration_df)


#Task 4
# Load data
returns = np.log(data.iloc[:, 1:] / data.iloc[:, 1:].shift(1))
returns = returns.dropna()

# Perform cointegration test
result = coint_johansen(returns, det_order=0, k_ar_diff=1)
eigenvalues = result.eig
critical_values = result.cvt[:, 1]  # Use 5% significance level
num_cointegrating = np.sum(eigenvalues > critical_values)

if num_cointegrating > 0:
    # Select a pair of cointegrated assets (e.g., the first two assets)
    asset1 = returns.columns[0]
    asset2 = returns.columns[1]

    # Estimate the Error Correction Model (ECM)
    model = VAR(returns[[asset1, asset2]])
    lag_order = model.select_order().selected_orders['bic']
    model_fitted = model.fit(lag_order)
    residuals = model_fitted.resid

    # Extract the cointegration vector
    coint_vector = result.evec[:, :num_cointegrating]

    # Estimate the ECM parameters
    lagged_diff = returns.diff().dropna()
    lagged_diff_coint = lagged_diff @ coint_vector
    lagged_ecm = lagged_diff_coint.shift(1).dropna()
    ecm_model = VAR(lagged_ecm)
    ecm_model_fitted = ecm_model.fit(lag_order)

    # Get the error correction coefficient (lambda)
    error_correction_coeff = ecm_model_fitted.coefs[0][0]

    # Print the results
    print("Cointegration Vector:")
    print(coint_vector)
    print("\nError Correction Coefficient (lambda):")
    print(error_correction_coeff)
else:
    print("No cointegration relationship found.")
