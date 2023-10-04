import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

data = pd.read_csv('concrete.csv')
data.dropna(inplace=True)

X = data[['cement (kg)', 'age (days)', 'water (kg)']]
y = data['strength (Mpa)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

coefficients = regression_model.coef_
intercept = regression_model.intercept_

coefficients_str = ' + '.join([f'{coef:.2f} * {var}' for coef, var in zip(coefficients, ['cement (kg)', 'age (days)', 'water (kg)'])])
equation = f'y = {intercept:.2f} + {coefficients_str}'

y_pred = regression_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Regression Equation:')
print(equation)
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R^2) Score: {r2:.2f}')
print('Coefficients:')
for var, coef in zip(['cement (kg)', 'age (days)', 'water (kg)'], coefficients):
    print(f'{var}: {coef:.2f}')

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

axes[0, 0].scatter(y_test, y_pred, c='blue')
axes[0, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
axes[0, 0].set_xlabel('Actual Strength (Mpa)')
axes[0, 0].set_ylabel('Predicted Strength (Mpa)')
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

residuals = y_test - y_pred
axes[0, 1].scatter(y_pred, residuals, c=residuals, cmap='viridis', alpha=0.6)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Fitted Values')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

axes[0, 2].hist(y_pred, bins=30, color='blue', alpha=0.7)
axes[0, 2].set_xlabel('Predicted Strength (Mpa)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].grid(True, linestyle='--', alpha=0.6)

axes[1, 0].hist(residuals, bins=30, color='green', alpha=0.7)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

stats.probplot(residuals, plot=axes[1, 1])
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

axes[1, 2].scatter(y_pred, residuals, c=residuals, cmap='viridis', alpha=0.6)
axes[1, 2].axhline(y=0, color='r', linestyle='--')
axes[1, 2].set_xlabel('Fitted Values')
axes[1, 2].set_ylabel('Residuals')
axes[1, 2].grid(True, linestyle='--', alpha=0.6)

plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9, hspace=0.5, wspace=0.4)  # Adjust layout

plt.tight_layout()
plt.show()
