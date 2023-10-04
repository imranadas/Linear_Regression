import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

# Load data
df = pd.read_csv('concrete.csv')
rand_df = df.sample(900)
X = rand_df['cement (kg)'].values
Y = rand_df['strength (Mpa)'].values

n = len(X)
X_mean = np.mean(X)
Y_mean = np.mean(Y)

beta1 = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean)**2)
beta0 = Y_mean - beta1 * X_mean

Y_pred = beta0 + beta1 * X

SSR = np.sum((Y_pred - Y_mean)**2)
SST = np.sum((Y - Y_mean)**2)
r2 = 1 - (SSR / SST)

# Descriptive Statistics
X_std = np.std(X)
Y_std = np.std(Y)

print(f'X Mean: {X_mean}, Y Mean: {Y_mean}')
print(f'X Standard Deviation: {X_std}, Y Standard Deviation: {Y_std}')
print(f'Regression Line: strength = {beta0:.2f} + {beta1:.2f}*cement')
print(f'Intercept (beta0): {beta0}')
print(f'Slope (beta1): {beta1}')
MSE = np.sum((Y_pred - Y)**2) / n
print(f'Mean Squared Error (MSE): {MSE}')
print(f'R^2 Score: {r2}')

# Residual Analysis
residuals = Y - Y_pred
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(residuals, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.subplot(1, 2, 2)
plt.scatter(Y_pred, residuals, s=10, color='coral')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.tight_layout()

# QQ Plot for Normality Check
sm.qqplot(residuals, line='s')
plt.title('QQ Plot')

# Influential Points
influence = OLSInfluence(sm.OLS(Y, sm.add_constant(X)).fit())
influential_points = np.abs(influence.resid_press) > 2 * np.mean(np.abs(influence.resid_press))
plt.figure()
plt.scatter(X, Y, s=10, color='skyblue', label='Data Points')
plt.scatter(X[influential_points], Y[influential_points], color='red', label='Influential Points')
plt.xlabel('Cement (KGs)')
plt.ylabel('Compressive Strength (MPa)')
plt.legend()
plt.title('Influential Points')

# Linear Regression Plot
gradient = (Y - Y_pred) / (Y.max() - Y.min())
plt.figure()
plt.scatter(X, Y, s=10, c=gradient, cmap='inferno_r', label='Data Points')
plt.plot(X, Y_pred, color='Black', linewidth=3, label='Regression Line')
plt.colorbar(label='Relative Strength')
plt.xlabel('Cement (KGs)')
plt.ylabel('Compressive Strength (MPa)')
plt.legend()
plt.title('Linear Regression')
plt.show()