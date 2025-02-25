import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from meteostat import Stations, Daily, Hourly
from datetime import datetime, timedelta

# ANSI color codes for better readability
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Centralized date definitions
START_DATE = datetime(2024, 1, 1)  # Example: Jan 1, 2024
END_DATE = datetime.today()  # Default to today's date

# Default future prediction period (7 days from today)
DEFAULT_FUTURE_DAYS = 7

# Allow manual future dates if needed (set to None to use default)
manual_dates = None  # Example: ["2025-02-20", "2025-02-21"]

# Generate future dates consistently
if manual_dates:
    future_dates = pd.to_datetime(manual_dates)
else:
    future_dates = pd.date_range(start=END_DATE, periods=DEFAULT_FUTURE_DAYS, freq='D')


# Get the Bet Dagan station dynamically
def get_bet_dagan_station():
    stations = Stations().region('IL').fetch(10)  # Get a list of stations
    bet_dagan = stations[stations['name'].str.contains("Bet Dagan", case=False, na=False)]
    
    if bet_dagan.empty:
        raise ValueError("❌ Bet Dagan station not found!")

    station_id = bet_dagan.index[0]  # Get the correct station ID
    print(f"✅ Using weather station: {bet_dagan.iloc[0]['name']} (ID: {station_id})")
    return station_id


def fetch_daily_data(station_id):
    data = Daily(station_id, start=START_DATE, end=END_DATE).fetch()
    if data.empty:
        raise ValueError("❌ No historical daily data found!")
    data.fillna(method='ffill', inplace=True)
    data['rain'] = (data['prcp'] > 0).astype(int)
    return data


def fetch_hourly_analysis(station_id):
    data = Hourly(station_id, start=START_DATE, end=END_DATE).fetch()
    if data.empty:
        raise ValueError("❌ No historical hourly data found!")

    data['hour'] = data.index.hour
    time_periods = {
        "Morning (6AM-12PM)": data[(data['hour'] >= 6) & (data['hour'] < 12)],
        "Afternoon (12PM-6PM)": data[(data['hour'] >= 12) & (data['hour'] < 18)],
        "Evening (6PM-12AM)": data[(data['hour'] >= 18) & (data['hour'] < 24)],
        "Night (12AM-6AM)": data[(data['hour'] >= 0) & (data['hour'] < 6)]
    }
    
    summary = {}
    for period, df in time_periods.items():
        if not df.empty:
            summary[period] = {
                "Avg Temp": round(df["temp"].mean(), 1),
                "Max Wind Speed": round(df["wspd"].max(), 1),
                "Total Rain": round(df["prcp"].sum(), 1)
            }
    
    return summary


# Train a model to predict rain
def train_model(data):
    required_features = ['tavg', 'tmin', 'tmax', 'prcp']
    
    if not all(col in data.columns for col in required_features):
        raise ValueError(f"❌ Missing required features: {set(required_features) - set(data.columns)}")

    X = data[required_features].fillna(data.mean())
    y = data['rain']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if X_train.empty:
        raise ValueError("❌ Training set is empty! Adjust test_size.")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print(GREEN + "✅ Model training complete." + RESET)
    return model


# Predict rain for the next few days and each part of the day
def predict_rain(model, data, hourly_data, future_days=7):
    today = datetime.today().date()
    future_dates = pd.date_range(start=today, periods=future_days, freq='D')

    # Predict rain for the next few days
    future_weather = pd.DataFrame(index=future_dates)

    # Use the latest available data to predict future weather
    latest_data = data.iloc[-1]  # Latest daily data
    for feature in ['tavg', 'tmin', 'tmax', 'prcp']: 
        future_weather[feature] = latest_data[feature]

    # Display rain predictions for each future day
    print(BLUE + "\n🌧️ Rain Prediction for Next Few Days:" + RESET)
    for future_date in future_dates:
        print(f"\nPrediction for {future_date.date()}:")
        # You can base the prediction on the average temperature or rainfall for the day
        rain_pred = model.predict(future_weather.loc[[future_date]])[0]
        color = GREEN if rain_pred == 0 else RED
        print(f"{color}Rain prediction: {'Rain' if rain_pred == 1 else 'No Rain'}{RESET}")

        # Now predict for each time period (morning, afternoon, etc.)
        print("\nBy Time of Day:")
        for period, stats in hourly_data.items():
            print(f"{period}:")
            # Predict rain based on the total rainfall in that time period
            rain_pred_period = 1 if stats['Total Rain'] > 0 else 0
            color_period = GREEN if rain_pred_period == 0 else RED
            print(f"{color_period}Rain prediction: {'Rain' if rain_pred_period == 1 else 'No Rain'}{RESET}")


# Display weather summary
def display_summary(data, hourly_analysis):
    print(YELLOW + "\n📊 Weather Summary (Past 7 Days & Next 7 Days):" + RESET)
    print(data[['tavg', 'tmin', 'tmax', 'prcp']].tail(14))  # Show past & future data

    print(BLUE + "\n🕒 Daily Averages by Time of Day (Past 7 Days):" + RESET)
    for period, stats in hourly_analysis.items():
        print(f"{period}: Temp={stats['Avg Temp']}°C, Wind={stats['Max Wind Speed']} km/h, Rain={stats['Total Rain']}mm")

# Run the script
if __name__ == "__main__":
    station_id = get_bet_dagan_station()
    daily_data = fetch_daily_data(station_id)
    hourly_analysis = fetch_hourly_analysis(station_id)

    display_summary(daily_data, hourly_analysis)
    
    model = train_model(daily_data)
    predict_rain(model, daily_data, hourly_analysis, future_days=7)  # Predict for next 7 days
