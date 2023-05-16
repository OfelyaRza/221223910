#Ofelya Rzayeva 221223910
#Exercise_5

#Load the necessary libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

#Task 1
#Import data
data = pd.read_csv("02_python_data.csv", index_col=0)
gdaxi_returns = np.log(data[".GDAXI"] / data[".GDAXI"].shift(1))
constituents_returns = np.log(data.iloc[:, 1:] / data.iloc[:, 1:].shift(1))

# Drop gdaxi values
gdaxi_returns = gdaxi_returns.dropna()
constituents_returns = constituents_returns.dropna()

correlation_matrix = np.corrcoef(constituents_returns.T)

n = len(constituents_returns.columns)  # Number of constituents
x0 = np.full(n, 1 / n)  # Initial guess for weights

# Define the objective function (portfolio variance)
def portfolio_variance(weights):
    return np.dot(np.dot(weights, correlation_matrix), weights)

# Define the equality constraint (sum of weights equals 1)
def weight_constraint(weights):
    return np.sum(weights) - 1

# Define the bounds for the weights (non-negative)
bounds = [(0, None)] * n

# Solve the optimization problem
constraints = [{'type': 'eq', 'fun': weight_constraint}]
result = minimize(portfolio_variance, x0, method='SLSQP', constraints=constraints, bounds=bounds)

# Extract the optimized weights
weights_mvp = result.x

#Calculate the risk (standard deviation) and return (average) of the DAX index and MVP
dax_risk = np.std(gdaxi_returns)
dax_return = np.mean(gdaxi_returns)

dax_avg_return = np.mean(gdaxi_returns)

mvp_risk = np.sqrt(portfolio_variance(weights_mvp))
mvp_return = np.mean(np.dot(constituents_returns, weights_mvp))

mvp_risk = np.sqrt(portfolio_variance(weights_mvp))
mvp_return = np.mean(np.dot(constituents_returns, weights_mvp))

#Plot the cumulative log-returns of both portfolios using matplotlib.
gdaxi_cum_returns = np.cumsum(gdaxi_returns)
mvp_cum_returns = np.cumsum(np.dot(constituents_returns, weights_mvp))
plt.plot(gdaxi_cum_returns, label=".GDAXI")
plt.plot(mvp_cum_returns, label="Minimum Variance Portfolio")
plt.title("Cumulative Log-Returns of .GDAXI and Minimum Variance Portfolio")
plt.xlabel("Date")
plt.ylabel("Cumulative Log-Returns")
plt.legend()
plt.show()

#Task 2

# Fit the linear regression model using all constituents
model = LinearRegression()
model.fit(constituents_returns, gdaxi_returns)
weights = model.coef_

#Calculate the cumulative log-returns of both portfolios:
portfolio_returns = np.dot(constituents_returns, weights)
dax_cumulative_return = np.cumsum(np.log(1 + gdaxi_returns))
portfolio_cumulative_returns = np.cumsum(np.log(1 + portfolio_returns))

#calculate tracking error
tracking_error = gdaxi_returns - portfolio_returns

#calculation prtfolio risk and return
portfolio_risk = np.std(portfolio_returns)
portfolio_return = np.mean(portfolio_returns)

#Plot the cumulative log-returns of both portfolios:
plt.plot(dax_cumulative_return, label='DAX Index')
plt.plot(portfolio_cumulative_returns, label='Replicating Portfolio')
plt.plot(tracking_error, label='Tracking Error')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Cumulative Log-Returns')
plt.title('DAX Index vs. Replicating Portfolio')
plt.show()

print("DAX Index:")
print("Risk (Standard Deviation):", dax_risk)
print("Return (Average):", dax_return)
print()
print("Replicating Portfolio:")
print("Risk (Standard Deviation):", portfolio_risk)
print("Return (Average):", portfolio_return)


#Task 3

# Select top 3 constituents with highest correlation with DAX index returns
top_3_constituents = constituents_returns.corrwith(gdaxi_returns).abs().nlargest(3).index

# Create a subset of constituents returns with the top 3 constituents
subset_constituents_returns = constituents_returns[top_3_constituents]

# Use linear regression to find the weights of the top 3 constituents
model = LinearRegression()
model.fit(subset_constituents_returns, gdaxi_returns)
weights_subset = model.coef_

# Calculate portfolio returns using the subset of constituents and their weights
portfolio_returns_subset = np.dot(subset_constituents_returns, weights_subset)

# Plot the cumulative log-returns and tracking error for the subset-based portfolio
subset_cumulative_returns = np.cumsum(np.log(1 + portfolio_returns_subset))
subset_tracking_error = gdaxi_returns - portfolio_returns_subset
plt.plot(dax_cumulative_return, label='DAX Index')
plt.plot(subset_cumulative_returns, label='Subset-Based Portfolio')
plt.plot(subset_tracking_error, label='Subset Tracking Error')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Cumulative Log-Returns')
plt.title('DAX Index vs. Subset-Based Portfolio')
plt.show()

print("Subset-Based Portfolio:")
print("Risk (Standard Deviation):", np.std(portfolio_returns_subset))
print("Return (Average):", np.mean(portfolio_returns_subset))
