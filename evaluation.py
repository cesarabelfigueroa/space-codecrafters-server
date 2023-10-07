import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load the model from the file
model = load('random_forest_model.joblib') 

# New data to predict
new_data = pd.DataFrame({
    'hour': [0],
    'day': [2],
    'month': [3],
    'year': [2024]
})

# Using the model to make predictions for the new data
predictions = model.predict(new_data)


# Displaying the predictions
print(f"Predicted Kp for 2024-03-02 00:00:00 is {predictions[0]:.2f}")