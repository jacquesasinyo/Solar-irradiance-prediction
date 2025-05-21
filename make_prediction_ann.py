import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

# Standardize features for Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create a scikit-learn compatible wrapper for the Keras model
# This will be used for feature importance calculation later
from sklearn.base import BaseEstimator, RegressorMixin

class KerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model=None):
        self.model = model
        
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        return self.model.predict(X).flatten()

# Create and train a neural network model
print("Training Neural Network model...")

# Define the model architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(1)  # Output layer (no activation for regression)
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'
)

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('ANN Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('ann_training_history.png')
plt.close()

# Filter for the target date
day_data = window_data[window_data['datetime'].dt.strftime('%Y-%m-%d') == target_date].copy()

if len(day_data) == 0:
    print(f"No data found for {target_date}")
else:
    # Make predictions
    X_pred = day_data[feature_cols]
    X_pred_scaled = scaler.transform(X_pred)  # Use the same scaler
    predictions = model.predict(X_pred_scaled).flatten()
    
    # Ensure predictions are non-negative
    predictions = np.maximum(predictions, 0)
    
    day_data['Predicted_GHI'] = predictions
    
    # Calculate error metrics
    rmse = np.sqrt(mean_squared_error(day_data['Global Horizontal [W/m^2]'], predictions))
    mae = mean_absolute_error(day_data['Global Horizontal [W/m^2]'], predictions)
    
    print(f"\nError metrics for {target_date}:")
    print(f"Neural Network RMSE: {rmse:.2f}")
    print(f"Neural Network MAE: {mae:.2f}")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    plt.plot(day_data['datetime'], day_data['Global Horizontal [W/m^2]'], 
             label='Actual', alpha=0.7, color='blue')
    plt.plot(day_data['datetime'], predictions, 
             label='ANN Predicted', alpha=0.7, color='purple')
    
    # Format x-axis to show hours
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    
    plt.title(f'Solar Irradiance Predictions (Neural Network) vs Actual Values {target_date}')
    plt.xlabel('Time of Day')
    plt.ylabel('Global Horizontal Irradiance (W/m²)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('solar_predictions_ann.png')
    plt.close()
    
    # Analyze feature importance for neural networks using permutation importance
    # This is a better approach for neural networks than trying to extract weights
    from sklearn.inspection import permutation_importance
    
    # Create a wrapped model instance that sklearn can use
    keras_regressor = KerasRegressor(model=model)
    
    # Use the wrapped model with permutation_importance
    result = permutation_importance(
        keras_regressor, 
        X_pred_scaled, 
        day_data['Global Horizontal [W/m^2]'].values,
        n_repeats=10,
        random_state=42
    )
    
    # Create feature importance dictionary
    feature_importance = {feature: score for feature, score in zip(feature_cols, result.importances_mean)}
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 feature importance (Neural Network, based on permutation importance):")
    for feat, score in sorted_importance[:10]:
        print(f"{feat}: {score:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame(sorted_importance, columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=True).tail(15)  # Top 15 features
    
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Permutation Importance')
    plt.ylabel('Feature')
    plt.title('Top 15 Feature Importance (Neural Network)')
    plt.tight_layout()
    plt.savefig('feature_importance_ann.png')
    plt.close()
    
    # Save predictions to CSV
    day_data[['datetime', 'Global Horizontal [W/m^2]', 'Predicted_GHI']].to_csv(
        f'ann_predictions_{target_date}.csv', index=False,
        columns=['datetime', 'Global Horizontal [W/m^2]', 'Predicted_GHI'],
        header=['Datetime', 'Actual_GHI', 'Predicted_GHI']
    )
    print(f"Predictions saved to 'ann_predictions_{target_date}.csv'")

    # Create a summary comparison plot with all models (needs to load results from other models)
    try:
        # Try to load XGBoost predictions if available
        xgb_predictions = pd.read_csv(f'xgboost_predictions_{target_date}.csv', parse_dates=['Datetime'])
        has_xgb = True
    except:
        has_xgb = False
        print("No XGBoost predictions file found for comparison")
    
    try:
        # Try to load Random Forest predictions if available
        rf_predictions = pd.read_csv(f'rf_predictions_{target_date}.csv', parse_dates=['Datetime'])
        has_rf = True
    except:
        has_rf = False
        print("No Random Forest predictions file found for comparison")
    
    if has_xgb or has_rf:
        plt.figure(figsize=(15, 8))
        plt.plot(day_data['datetime'], day_data['Global Horizontal [W/m^2]'], 
                label='Actual', alpha=0.9, color='blue', linewidth=2)
        
        # Add ANN predictions
        plt.plot(day_data['datetime'], predictions, 
                label='ANN Predicted', alpha=0.7, color='purple', linewidth=1.5)
        
        # Add XGBoost predictions if available
        if has_xgb:
            plt.plot(xgb_predictions['Datetime'], xgb_predictions['Predicted_GHI'], 
                    label='XGBoost Predicted', alpha=0.7, color='red', linewidth=1.5)
        
        # Add Random Forest predictions if available
        if has_rf:
            plt.plot(rf_predictions['Datetime'], rf_predictions['Predicted_GHI'], 
                    label='RF Predicted', alpha=0.7, color='green', linewidth=1.5)
        
        # Format x-axis to show hours
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        
        plt.title(f'Comparison of Model Predictions for {target_date}')
        plt.xlabel('Time of Day')
        plt.ylabel('Global Horizontal Irradiance (W/m²)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the comparison plot
        plt.savefig('model_comparison_plot.png')
        plt.close()
        print("Comparison plot saved as 'model_comparison_plot.png'")