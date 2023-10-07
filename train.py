import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

# Data Loading
data = pd.read_csv("data.csv")
data['datetime'] = pd.to_datetime(data['datetime'])
data['hour'] = data['datetime'].dt.hour
data['day'] = data['datetime'].dt.day
data['month'] = data['datetime'].dt.month
data['year'] = data['datetime'].dt.year

# Selecting Features and Target
features = ['hour', 'day', 'month', 'year']
X = data[features]
y = data['Kp']

# Data Splitting
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

# Model Training
model = RandomForestRegressor(random_state=0)
model.fit(X_train, y_train)

# Storing Model
dump(model, 'random_forest_model.joblib') 
