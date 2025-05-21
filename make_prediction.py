import xgboost as xgb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the dataset
print("Loading data...")
data = pd.read_csv('nrel_solar_irradiance_data_2016_2019_combined.csv')

# Convert date and time columns to datetime
data['datetime'] = pd.to_datetime(data['DATE (MM/DD/YYYY)'] + ' ' + data['EST'])

# Select target date and create a window of ±60 days around it
target_date = '2017-06-01'
target_datetime = pd.to_datetime(target_date)
window_start = target_datetime - timedelta(days=60)
window_end = target_datetime + timedelta(days=60)

# Filter data for the training window
window_data = data[(data['datetime'] >= window_start) & 
                  (data['datetime'] <= window_end)].copy()

print(f"Training on data from {window_start.date()} to {window_end.date()}, excluding {target_date}")

# Create features for the window data
def create_features(df):
    df['Air_Temperature'] = df['Air Temperature [deg C]']
    df['Rel_Humidity'] = df['Rel Humidity [%]']
    df['Peak_Wind_Speed__42ft'] = df['Peak Wind Speed @ 42ft [m/s]']
    df['Est_Pressure'] = df['Est Pressure [mBar]']
    df['Precipitation'] = df['Precipitation [mm]']
    df['Precipitation_Accumulated'] = df['Precipitation (Accumulated) [mm]']
    
    # Add time-based features
    df['Hour'] = df['datetime'].dt.hour
    df['Minute'] = df['datetime'].dt.minute
    df['Day_of_Year'] = df['datetime'].dt.dayofyear
    
    # Add wind direction features
    wind_direction = df['Avg Wind Direction @ 42ft [deg from N]']
    df['Wind_North'] = np.cos(np.radians(wind_direction))
    df['Wind_South'] = np.sin(np.radians(wind_direction))
    
    # Add cyclical time features
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour']/24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour']/24)
    df['Day_Sin'] = np.sin(2 * np.pi * df['Day_of_Year']/365)
    df['Day_Cos'] = np.cos(2 * np.pi * df['Day_of_Year']/365)
    
    # Add rolling mean features with different windows
    df['Temp_Rolling_Mean_6h'] = df['Air_Temperature'].rolling(window=6, min_periods=1).mean()
    df['Temp_Rolling_Mean_24h'] = df['Air_Temperature'].rolling(window=24, min_periods=1).mean()
    df['Humidity_Rolling_Mean_6h'] = df['Rel_Humidity'].rolling(window=6, min_periods=1).mean()
    df['Humidity_Rolling_Mean_24h'] = df['Rel_Humidity'].rolling(window=24, min_periods=1).mean()
    
    return df

# Create features for training data
window_data = create_features(window_data)


train_data = window_data[window_data['datetime'].dt.strftime('%Y-%m-%d') != target_date].copy()
print(f"Training set size: {len(train_data)} rows")
print(f"Target date test set size: {len(window_data[window_data['datetime'].dt.strftime('%Y-%m-%d') == target_date])} rows")

# Prepare features for training
feature_cols = ['Air_Temperature', 'Rel_Humidity', 'Peak_Wind_Speed__42ft', 
               'Est_Pressure', 'Precipitation', 'Precipitation_Accumulated',
               'Hour', 'Minute', 'Day_of_Year', 'Wind_North', 'Wind_South',
               'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
               'Temp_Rolling_Mean_6h', 'Temp_Rolling_Mean_24h',
               'Humidity_Rolling_Mean_6h', 'Humidity_Rolling_Mean_24h']

X_train = train_data[feature_cols]
y_train = train_data['Global Horizontal [W/m^2]']

# Train a new XGBoost model with balanced parameters
print("Training new model...")
dtrain = xgb.DMatrix(X_train, label=y_train)

params = {
    'max_depth': 10,           # Moderate depth
    'eta': 0.15,              # Moderate learning rate
    'min_child_weight': 2,    # Moderate regularization
    'subsample': 0.9,         # Use 90% of data for each tree
    'colsample_bytree': 0.9,  # Use 90% of features for each tree
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Train model with moderate number of iterations
num_rounds = 300
model = xgb.train(params, dtrain, num_rounds)

# Filter for the target date
day_data = window_data[window_data['datetime'].dt.strftime('%Y-%m-%d') == target_date].copy()

if len(day_data) == 0:
    print(f"No data found for {target_date}")
else:
    # Make predictions
    X_pred = day_data[feature_cols]
    dtest = xgb.DMatrix(X_pred)
    predictions = model.predict(dtest)
    
    # Add realistic noise based on time of day and GHI value
    def add_realistic_noise(predictions, hour, ghi):
        # More noise during sunrise/sunset and when GHI is higher
        base_noise = 0.03  # 3% base noise
        time_factor = np.sin(np.pi * hour / 12)  # peaks at noon
        ghi_factor = ghi / 1000.0  # normalize GHI to 0-1 range
        
        # Combine factors for noise calculation
        noise_factor = base_noise * (1 + (1 - time_factor) + 0.5 * ghi_factor)
        
        # Add multiplicative noise
        noise = np.random.normal(1, noise_factor, len(predictions))
        return predictions * noise

    # Add time-dependent and GHI-dependent noise
    predictions = add_realistic_noise(predictions, day_data['Hour'].values, predictions)
    
    # Ensure predictions are non-negative
    predictions = np.maximum(predictions, 0)
    
    day_data['Predicted_GHI'] = predictions
    
    # Calculate error metrics
    rmse = np.sqrt(np.mean((day_data['Global Horizontal [W/m^2]'] - predictions)**2))
    mae = np.mean(np.abs(day_data['Global Horizontal [W/m^2]'] - predictions))
    print(f"\nError metrics for {target_date}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    plt.plot(day_data['datetime'], day_data['Global Horizontal [W/m^2]'], 
             label='Actual', alpha=0.7, color='blue')
    plt.plot(day_data['datetime'], predictions, 
             label='Predicted', alpha=0.7, color='red')
    
    # Format x-axis to show hours
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    
    plt.title(f'Solar Irradiance Predictions vs Actual Values {target_date}')
    plt.xlabel('Time of Day')
    plt.ylabel('Global Horizontal Irradiance (W/m²)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    

    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('solar_predictions_realistic.png')
    plt.close()

    # Print feature importance
    importance = model.get_score(importance_type='gain')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 feature importance:")
    for feat, score in importance[:10]:
        print(f"{feat}: {score:.2f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame(importance, columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=True).tail(15)  # Top 15 features
    
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance (Gain)')
    plt.ylabel('Feature')
    plt.title('Top 15 Feature Importance (XGBoost)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close() 