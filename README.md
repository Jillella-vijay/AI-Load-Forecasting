# AI-Load-Forecasting
# AI-Based Smart Load Forecasting for Power Distribution

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load Dataset
# (You can replace this with real dataset)
data = pd.read_csv('load_data.csv')

# Step 2: Preprocessing
data = data.dropna()

# Features (Example)
X = data[['Temperature', 'Humidity', 'WindSpeed']]
y = data['Load']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Prediction
predictions = model.predict(X_test)

# Step 6: Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE:", mae)
print("RMSE:", rmse)

# Step 7: Visualization
plt.plot(y_test.values, label='Actual Load')
plt.plot(predictions, label='Predicted Load')
plt.legend()
plt.title("Load Forecasting")
plt.show()
