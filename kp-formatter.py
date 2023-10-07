import pandas as pd
import json

with open("unformatted_data.json", "r") as file:
    data = json.load(file)

df = pd.DataFrame({
    'datetime': data['datetime'],
    'Kp': data['Kp'],
    'status': data['status']
})
df['datetime'] = pd.to_datetime(df['datetime'])  
df['Kp_diff'] = df['Kp'].diff() 
df.to_csv('data.csv', index=False)





print(df[['datetime', 'Kp', 'Kp_diff']])
