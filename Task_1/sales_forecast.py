import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("sales.csv")

# Convert date column
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values('date')

# Feature Engineering
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['day_of_week'] = df['date'].dt.dayofweek

# Lag feature (previous day sales)
df['lag1'] = df['sales'].shift(1)

# Remove null values
df = df.dropna()

# Features & target
X = df[['day','month','year','day_of_week','lag1','holiday','promo']]
y = df['sales']

# Train-test split
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, pred)
print("Mean Absolute Error:", mae)

# Plot Actual vs Predicted
plt.figure(figsize=(10,5))
plt.plot(df['date'][split:], y_test, label='Actual')
plt.plot(df['date'][split:], pred, label='Predicted')
plt.legend()
plt.title("Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# Future Forecast (next 5 days)
future_dates = pd.date_range(start=df['date'].max(), periods=6)[1:]

future_df = pd.DataFrame({'date': future_dates})
future_df['day'] = future_df['date'].dt.day
future_df['month'] = future_df['date'].dt.month
future_df['year'] = future_df['date'].dt.year
future_df['day_of_week'] = future_df['date'].dt.dayofweek

# Use last known sales value
last_sales = df['sales'].iloc[-1]
future_df['lag1'] = last_sales

# IMPORTANT FIX: add missing features
future_df['holiday'] = 0
future_df['promo'] = 0

# Predict future
future_pred = model.predict(
    future_df[['day','month','year','day_of_week','lag1','holiday','promo']]
)

# Plot future forecast
plt.figure(figsize=(10,5))
plt.plot(df['date'], df['sales'], label='Historical')
plt.plot(future_df['date'], future_pred, label='Forecast', linestyle='dashed')
plt.legend()
plt.title("Future Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()
