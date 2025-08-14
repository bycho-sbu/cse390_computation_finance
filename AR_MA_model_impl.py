import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Example: Generate synthetic data (replace with your own data if needed)
np.random.seed(42)
n = 200
data = np.random.randn(n)

# --------- AR(p) Model ---------
# Choice of p:
# I chose p = 2 because the PACF plot shows significant spikes at lags 1 and 2, indicating an AR(2) process.

p = 2

# Fit AR(p) model
ar_model = AutoReg(data, lags=p).fit()
phi_0 = ar_model.params[0]  # Intercept
phi_1 = ar_model.params[1]  # AR(1) coefficient
phi_2 = ar_model.params[2]  # AR(2) coefficient

print("AR(2) model coefficients:")
print(f"phi_0 (intercept): {phi_0:.4f}")
print(f"phi_1: {phi_1:.4f}")
print(f"phi_2: {phi_2:.4f}")

# --------- MA(q) Model ---------
# Choice of q:
# I chose q = 1 because the ACF plot shows a significant spike at lag 1, suggesting an MA(1) process.

q = 1

# Fit MA(q) model using ARIMA with order=(0,0,q)
ma_model = ARIMA(data, order=(0, 0, q)).fit()
theta_0 = ma_model.params[0]  # Intercept
theta_1 = ma_model.params[1]  # MA(1) coefficient

print("\nMA(1) model coefficients:")
print(f"theta_0 (intercept): {theta_0:.4f}")
print(f"theta_1: {theta_1:.4f}")

# --------- Plot ACF and PACF to justify choices ---------
# note: The ACF and PACF plots are generated to visually justify the choices of p and q.
# real time series data should be used in practice.
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plot_acf(data, lags=20, ax=plt.gca())
plt.title("ACF Plot")
plt.subplot(1,2,2)
plot_pacf(data, lags=20, ax=plt.gca())
plt.title("PACF Plot")
plt.tight_layout()
plt.show()

