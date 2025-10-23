
from fastapi import FastAPI
import yfinance as yf

from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://trading-peach-five.vercel.app"],  # Allows only the frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def read_root():
    return {"message": "Backend v1.1 - Hello World"}

@app.get("/predict")
def predict_stock(ticker: str):
    print(f"Received ticker: {ticker}")
    # Fetch data
    data = yf.download(ticker, start="2020-01-01", end="2025-10-23")
    if data.empty:
        return {"error": "Invalid stock symbol"}

    # Flatten multi-level columns from yfinance
    # Select only the 'Close' column and rename it
    data = data[('Close', ticker)]
    data = data.to_frame(name='Close') # Convert Series to DataFrame and name the column

    # Preprocess data
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date_ordinal'] = data['Date'].apply(lambda date: date.toordinal())

    X = data[['Date_ordinal']]
    y = data['Close']

    # Create and train model
    model = LinearRegression()
    model.fit(X, y)

    # Make prediction
    last_date_ordinal = data['Date_ordinal'].iloc[-1]
    prediction_date_ordinal = last_date_ordinal + 1
    prediction = model.predict([[prediction_date_ordinal]])
    # Recommendation Logic
    last_price = data['Close'].iloc[-1].item()
    if prediction[0] > last_price * 1.01:
        recommendation = "BUY"
    elif prediction[0] < last_price * 0.99:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    
    # Prepare data for charting
    chart_data = data[['Date', 'Close']].copy()
    
    # Create a dataframe for the prediction
    prediction_date = data['Date'].iloc[-1] + pd.Timedelta(days=1)
    prediction_df = pd.DataFrame({'Date': [prediction_date], 'Prediction': [prediction[0]]})
    
    # Combine with historical data for charting
    chart_data = pd.concat([chart_data, prediction_df], ignore_index=True)

    return {
        "stock_symbol": ticker,
        "prediction": prediction[0],
        "last_price": last_price,
        "recommendation": recommendation,
        "chart_data": {
            "dates": chart_data['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "prices": chart_data['Close'].replace({pd.NA: None, np.nan: None}).tolist(),
            "prediction": chart_data['Prediction'].replace({pd.NA: None, np.nan: None}).tolist()
        }
    }
