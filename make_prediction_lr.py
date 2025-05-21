import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

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
    
    # Add interaction features
    df['Temp_Humidity_Interaction'] = df['Air_Temperature'] * df['Rel_Humidity']
    df['Hour_Sin_Day_Sin_Interaction'] = df['Hour_Sin'] * df['Day_Sin']
    
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
               'Humidity_Rolling_Mean_6h', 'Humidity_Rolling_Mean_24h',
               'Temp_Humidity_Interaction', 'Hour_Sin_Day_Sin_Interaction']

X_train = train_data[feature_cols]
y_train = train_data['Global Horizontal [W/m^2]']

# Standardize features for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create a polynomial features transformer to capture non-linear relationships
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_train_poly = poly.fit_transform(X_train_scaled)

print(f"Original feature shape: {X_train_scaled.shape}")
print(f"Polynomial feature shape: {X_train_poly.shape}")

# Train a Linear Regression model
print("Training Linear Regression model...")
lr_model = LinearRegression(n_jobs=-1)
lr_model.fit(X_train_poly, y_train)

# Filter for the target date
day_data = window_data[window_data['datetime'].dt.strftime('%Y-%m-%d') == target_date].copy()

if len(day_data) == 0:
    print(f"No data found for {target_date}")
else:
    # Make predictions
    X_pred = day_data[feature_cols]
    X_pred_scaled = scaler.transform(X_pred)
    X_pred_poly = poly.transform(X_pred_scaled)
    predictions = lr_model.predict(X_pred_poly)
    
    # Ensure predictions are non-negative
    predictions = np.maximum(predictions, 0)
    
    day_data['Predicted_GHI'] = predictions
    
    # Calculate error metrics
    rmse = np.sqrt(mean_squared_error(day_data['Global Horizontal [W/m^2]'], predictions))
    mae = mean_absolute_error(day_data['Global Horizontal [W/m^2]'], predictions)
    r2 = r2_score(day_data['Global Horizontal [W/m^2]'], predictions)
    
    print(f"\nError metrics for {target_date}:")
    print(f"Linear Regression RMSE: {rmse:.2f}")
    print(f"Linear Regression MAE: {mae:.2f}")
    print(f"Linear Regression R²: {r2:.4f}")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    plt.plot(day_data['datetime'], day_data['Global Horizontal [W/m^2]'], 
             label='Actual', alpha=0.7, color='blue')
    plt.plot(day_data['datetime'], predictions, 
             label='LR Predicted', alpha=0.7, color='purple')
    
    # Format x-axis to show hours
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    
    plt.title(f'Solar Irradiance Predictions (Linear Regression) vs Actual Values {target_date}')
    plt.xlabel('Time of Day')
    plt.ylabel('Global Horizontal Irradiance (W/m²)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('solar_predictions_lr.png')
    plt.close()
    
    # Analyze feature importance for linear regression using coefficients and permutation importance
    # Get feature names after polynomial transformation
    feature_names = poly.get_feature_names_out(feature_cols)
    
    # Get coefficients
    coefficients = np.abs(lr_model.coef_)
    
    # Create DataFrame for coefficients
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    
    # Sort by absolute coefficient value
    coef_df = coef_df.sort_values('Coefficient', ascending=False)
    
    # Print top coefficients
    print("\nTop 15 features by coefficient magnitude:")
    print(coef_df.head(15))
    
    # Permutation importance (more reliable for feature importance)
    perm_importance = permutation_importance(
        lr_model, X_pred_poly, day_data['Global Horizontal [W/m^2]'], 
        n_repeats=10, random_state=42
    )
    
    # Create a DataFrame for permutation importances
    perm_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean
    })
    
    # Sort by importance
    perm_importance_df = perm_importance_df.sort_values('Importance', ascending=False)
    
    # Print permutation importances
    print("\nTop 15 features by permutation importance:")
    print(perm_importance_df.head(15))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importance_df = perm_importance_df.head(15).sort_values('Importance')
    
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Permutation Importance')
    plt.ylabel('Feature')
    plt.title('Top 15 Feature Importance (Linear Regression)')
    plt.tight_layout()
    plt.savefig('feature_importance_lr.png')
    plt.close()
    
    # Save predictions to CSV
    day_data[['datetime', 'Global Horizontal [W/m^2]', 'Predicted_GHI']].to_csv(
        f'lr_predictions_{target_date}.csv', index=False,
        columns=['datetime', 'Global Horizontal [W/m^2]', 'Predicted_GHI'],
        header=['Datetime', 'Actual_GHI', 'Predicted_GHI']
    )
    print(f"Predictions saved to 'lr_predictions_{target_date}.csv'")

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
    
    try:
        # Try to load ANN predictions if available
        ann_predictions = pd.read_csv(f'ann_predictions_{target_date}.csv', parse_dates=['Datetime'])
        has_ann = True
    except:
        has_ann = False
        print("No ANN predictions file found for comparison")
    
    try:
        # Try to load KNN predictions if available
        knn_predictions = pd.read_csv(f'knn_predictions_{target_date}.csv', parse_dates=['Datetime'])
        has_knn = True
    except:
        has_knn = False
        print("No KNN predictions file found for comparison")
    
    if has_xgb or has_rf or has_ann or has_knn:
        plt.figure(figsize=(15, 8))
        plt.plot(day_data['datetime'], day_data['Global Horizontal [W/m^2]'], 
                label='Actual', alpha=0.9, color='blue', linewidth=2)
        
        # Add LR predictions
        plt.plot(day_data['datetime'], predictions, 
                label='LR Predicted', alpha=0.7, color='purple', linewidth=1.5)
        
        # Add XGBoost predictions if available
        if has_xgb:
            plt.plot(xgb_predictions['Datetime'], xgb_predictions['Predicted_GHI'], 
                    label='XGBoost Predicted', alpha=0.7, color='red', linewidth=1.5)
        
        # Add Random Forest predictions if available
        if has_rf:
            plt.plot(rf_predictions['Datetime'], rf_predictions['Predicted_GHI'], 
                    label='RF Predicted', alpha=0.7, color='orange', linewidth=1.5)
        
        # Add ANN predictions if available
        if has_ann:
            plt.plot(ann_predictions['Datetime'], ann_predictions['Predicted_GHI'], 
                    label='ANN Predicted', alpha=0.7, color='green', linewidth=1.5)
        
        # Add KNN predictions if available
        if has_knn:
            plt.plot(knn_predictions['Datetime'], knn_predictions['Predicted_GHI'], 
                    label='KNN Predicted', alpha=0.7, color='cyan', linewidth=1.5)
        
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
        plt.savefig('model_comparison_plot_with_lr.png')
        plt.close()
        print("Comparison plot saved as 'model_comparison_plot_with_lr.png'")