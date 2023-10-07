import pandas as pd
from joblib import load
from fastapi import FastAPI



app = FastAPI()

@app.get("/predict_kp")
def predict_kp(start, end):
    # Load the model
    model = load('kp_model.joblib')

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
    return {"kp_and_dates": kp_and_dates}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    