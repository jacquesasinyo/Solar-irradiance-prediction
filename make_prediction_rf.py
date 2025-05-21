import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# Train a Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=300,        # Number of trees
    max_depth=10,            # Maximum depth of trees
    min_samples_split=5,     # Minimum samples required to split
    min_samples_leaf=2,      # Minimum samples at leaf node
    max_features='sqrt',     # Use sqrt(n_features) features in each tree
    bootstrap=True,          # Use bootstrap samples
    random_state=42,         # For reproducibility
    n_jobs=-1                # Use all available cores
)

rf_model.fit(X_train, y_train)

# Filter for the target date
day_data = window_data[window_data['datetime'].dt.strftime('%Y-%m-%d') == target_date].copy()

if len(day_data) == 0:
    print(f"No data found for {target_date}")
else:
    # Make predictions
    X_pred = day_data[feature_cols]
    predictions = rf_model.predict(X_pred)
    
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
    rmse = np.sqrt(mean_squared_error(day_data['Global Horizontal [W/m^2]'], predictions))
    mae = mean_absolute_error(day_data['Global Horizontal [W/m^2]'], predictions)
    print(f"\nError metrics for {target_date}:")
    print(f"Random Forest RMSE: {rmse:.2f}")
    print(f"Random Forest MAE: {mae:.2f}")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    plt.plot(day_data['datetime'], day_data['Global Horizontal [W/m^2]'], 
             label='Actual', alpha=0.7, color='blue')
    plt.plot(day_data['datetime'], predictions, 
             label='RF Predicted', alpha=0.7, color='green')
    
    # Format x-axis to show hours
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    
    plt.title(f'Solar Irradiance Predictions (Random Forest) vs Actual Values {target_date}')
    plt.xlabel('Time of Day')
    plt.ylabel('Global Horizontal Irradiance (W/m²)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('solar_predictions_rf.png')
    plt.close()

    # Print feature importance
    importance = rf_model.feature_importances_
    feature_importance = {feature: score for feature, score in zip(feature_cols, importance)}
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 feature importance (Random Forest):")
    for feat, score in sorted_importance[:10]:
        print(f"{feat}: {score:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame(sorted_importance, columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=True).tail(15)  # Top 15 features
    
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 15 Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png')
    plt.close()
    
    # Save predictions to CSV
    day_data[['datetime', 'Global Horizontal [W/m^2]', 'Predicted_GHI']].to_csv(
        f'rf_predictions_{target_date}.csv', index=False,
        columns=['datetime', 'Global Horizontal [W/m^2]', 'Predicted_GHI'],
        header=['Datetime', 'Actual_GHI', 'Predicted_GHI']
    )
    print(f"Predictions saved to 'rf_predictions_{target_date}.csv'")