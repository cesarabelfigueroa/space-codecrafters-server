import pandas as pd
from joblib import load
from fastapi import FastAPI
from datetime import datetime




app = FastAPI()

@app.get("/predict_kp")
def predict_kp(start, end):
    # Load the model
    result = {
        "G1": [], 
        "G2": [],
        "G3": [],
        "G4": []
    }
    model = load('kp_model.joblib')
    map = {
        "G1": [5.0,6.0], 
        "G2": [6.0, 7.0],
        "G3": [7.0, 8.0],
        "G4": [8.0, 9.0]
    }


    # Create a DataFrame for future predictions
    future_dates = pd.date_range(start=start, end=end, freq='H')
    future_data = pd.DataFrame({
        'hour': future_dates.hour,
        'day': future_dates.day,
        'month': future_dates.month,
        'year': future_dates.year
    })

    # Making predictions
    predictions = model.predict(future_data)

    # Identify the indexes and dates where the predicted Kp is >= 5
    indexes_kp_5_or_more = [i for i, kp in enumerate(predictions) if kp >= 5]
    dates_kp_5_or_more = future_dates[indexes_kp_5_or_more]
    kp_values_5_or_more = predictions[indexes_kp_5_or_more]

    # Output
    kp_and_dates = list(zip(dates_kp_5_or_more, kp_values_5_or_more))
    for score in map:
        for date, kp in kp_and_dates:
            if(kp >= map[score][0] and kp < map[score][1]):
                original_date = datetime.fromisoformat(str(date))
                formatted_date = original_date.strftime("%Y-%m-%d")
                result[score].append({"kp":kp, "date":formatted_date})

        df = pd.DataFrame(data=result[score], columns=['date', 'kp'])
        df.drop_duplicates(subset="date",inplace=True)
        result[score] = df.values
            
    return result





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    