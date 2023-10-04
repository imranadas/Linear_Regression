import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

data = pd.read_csv('concrete.csv')
data.dropna(inplace=True)

X = data[['cement (kg)', 'coarseagg (kg)']]
y = data['strength (Mpa)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

coefficients = regression_model.coef_
intercept = regression_model.intercept_

coefficients_str = ' + '.join([f'{coef:.2f} * {var}' for coef, var in zip(coefficients, ['cement (kg)', 'coarseagg (kg)'])])
equation = f'y = {intercept:.2f} + {coefficients_str}'

y_pred = regression_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Coefficients:')
for var, coef in zip(['cement (kg)', 'coarseagg (kg)'], coefficients):
    print(f'{var}: {coef:.2f}')
print('Regression Equation:')
print(equation)
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R^2) Score: {r2:.2f}')

plt.figure(figsize=(18, 8))

# Regression Plot
plt.subplot(2, 3, 1)
sns.regplot(x=y_test, y=regression_model.predict(X_test), line_kws={'color': 'blue'})
plt.xlabel('Actual Strength (Mpa)')
plt.ylabel('Predicted Strength (Mpa)')
plt.title('Regression Plot')
plt.plot(y_test, y_test, color='red', linestyle='--')  # Add 45-degree line
plt.grid(True, linestyle='--', alpha=0.6)

# Residuals Plot
plt.subplot(2, 3, 2)
residuals = y_test - y_pred
cmap = plt.get_cmap('viridis')
normalize = plt.Normalize(vmin=residuals.min(), vmax=residuals.max())
colors = cmap(normalize(residuals))
sc = plt.scatter(y_pred, residuals, c=colors, alpha=0.6)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.colorbar(sc, label='Residuals')  # Add colorbar
plt.grid(True, linestyle='--', alpha=0.6)

# Histogram of Predicted Values
plt.subplot(2, 3, 4)
plt.hist(y_pred, bins=20, color='skyblue', alpha=0.7)
plt.xlabel('Predicted Strength (Mpa)')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Values')
plt.grid(True, linestyle='--', alpha=0.6)

# Histogram of Residuals
plt.subplot(2, 3, 5)
plt.hist(residuals, bins=20, color='salmon', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.grid(True, linestyle='--', alpha=0.6)

# Q-Q Plot of Residuals
plt.subplot(2, 3, 3)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.grid(True, linestyle='--', alpha=0.6)

# Homoscedasticity Check
plt.subplot(2, 3, 6)
plt.scatter(y_pred, residuals, alpha=0.6)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Homoscedasticity Check')
plt.axhline(y=0, color='r', linestyle='--')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
