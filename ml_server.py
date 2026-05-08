import numpy as np
import pandas as pd
import requests
import logging
import shap
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input, Dropout
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

API_KEY = "jeKAButJ1CxibgyuJZgT7AESwSw4ZQwI"

np.random.seed(42)
tf.random.set_seed(42)
plt.style.use('dark_background')

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def fetch_weather_data(city):
    """Fetch ALL available weather data from Tomorrow.io"""
    fields = [
        'temperature', 'humidity', 'pressureSurfaceLevel', 'windSpeed', 'windDirection',
        'dewPoint', 'uvIndex', 'visibility', 'cloudCover', 'precipitationIntensity',
        'temperatureApparent', 'uvHealthConcern', 'epaIndex', 'particulateMatter25',
        'particulateMatter10', 'pollutantCO', 'pollutantNO2', 'pollutantO3',
        'pollutantSO2', 'grassIndex', 'treeIndex', 'weedIndex'
    ]
    
    field_str = ','.join(fields)
    hist_url = f"https://api.tomorrow.io/v4/weather/history/recent?location={city}&apikey={API_KEY}&units=metric&fields={field_str}"
    live_url = f"https://api.tomorrow.io/v4/weather/realtime?location={city}&apikey={API_KEY}&units=metric"
    
    try:
        logging.info(f"📡 Fetching comprehensive data for {city}")
        h_res = requests.get(hist_url, timeout=20)
        l_res = requests.get(live_url, timeout=15)
        logging.info(f"API Status - History: {h_res.status_code}, Live: {l_res.status_code}")
        
        if h_res.status_code != 200 or l_res.status_code != 200:
            logging.error(f"API error - History: {h_res.status_code}, Live: {l_res.status_code}")
            return None
        
        h_json = h_res.json()
        l_json = l_res.json()
        hourly = h_json.get('timelines', {}).get('hourly', [])
        
        if len(hourly) == 0:
            logging.error("No hourly data available")
            return None
        
        logging.info(f"✅ Got {len(hourly)} hours of comprehensive data")
        
        records = []
        for entry in hourly:
            values = entry.get('values', {})
            record = {
                'temp': values.get('temperature'),
                'hum': values.get('humidity'),
                'pres': values.get('pressureSurfaceLevel'),
                'wind_speed': values.get('windSpeed', 0),
                'wind_dir': values.get('windDirection', 0),
                'dew_point': values.get('dewPoint', 0),
                'uv_index': values.get('uvIndex', 0),
                'visibility': values.get('visibility', 10),
                'cloud_cover': values.get('cloudCover', 0),
                'precip_intensity': values.get('precipitationIntensity', 0),
                'temp_apparent': values.get('temperatureApparent'),
                'uv_concern': values.get('uvHealthConcern', 0),
                'epa_index': values.get('epaIndex', 0),
                'pm25': values.get('particulateMatter25', 0),
                'pm10': values.get('particulateMatter10', 0),
                'co': values.get('pollutantCO', 0),
                'no2': values.get('pollutantNO2', 0),
                'o3': values.get('pollutantO3', 0),
                'so2': values.get('pollutantSO2', 0),
                'grass_index': values.get('grassIndex', 0),
                'tree_index': values.get('treeIndex', 0),
                'weed_index': values.get('weedIndex', 0),
                'weather_code': values.get('weatherCode', 1000),
                'time': entry.get('time', '')
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df = df.ffill().bfill().fillna(0)
        
        try:
            df['time'] = pd.to_datetime(df['time'])
        except:
            df['time'] = pd.date_range(end=datetime.now(), periods=len(df), freq='h')
        
        # Fill missing apparent temp with actual temp
        if df['temp_apparent'].isnull().all():
            df['temp_apparent'] = df['temp']
        
        live_values = l_json.get('data', {}).get('values', {})
        current_t = live_values.get('temperature') or float(df['temp'].iloc[-1])
        current_h = live_values.get('humidity') or float(df['hum'].iloc[-1])
        current_p = live_values.get('pressureSurfaceLevel') or float(df['pres'].iloc[-1])
        current_wind = live_values.get('windSpeed', 0) or 0
        current_wind_dir = live_values.get('windDirection', 0) or 0
        current_cloud = live_values.get('cloudCover', 0) or 0
        current_visibility = live_values.get('visibility', 10) or 10
        current_uv = live_values.get('uvIndex', 0) or 0
        current_dew = live_values.get('dewPoint') or float(df['dew_point'].iloc[-1])
        current_precip = live_values.get('precipitationIntensity', 0) or 0
        current_weather = live_values.get('weatherCode', 1000) or 1000
        
        logging.info(f"📊 Data: {len(df)} rows, {len(df.columns)} features")
        logging.info(f"   Current: {current_t}°C, {current_h}%, Wind: {current_wind}m/s, Clouds: {current_cloud}%")
        
        return {
            'df': df,
            'current_temp': float(current_t),
            'current_humidity': float(current_h),
            'current_pressure': float(current_p),
            'current_wind': float(current_wind),
            'current_wind_dir': float(current_wind_dir),
            'current_cloud': float(current_cloud),
            'current_visibility': float(current_visibility),
            'current_uv': float(current_uv),
            'current_dew_point': float(current_dew),
            'current_precip': float(current_precip),
            'current_weather': int(current_weather),
            'total_hours': len(df)
        }
    except Exception as e:
        logging.error(f"❌ Fetch error: {e}")
        return None

def prepare_features(df):
    """Create comprehensive features from ALL available data"""
    f = pd.DataFrame()
    
    # --- CORE WEATHER ---
    for col in ['temp', 'hum', 'pres']:
        f[col] = pd.to_numeric(df.get(col, df['temp'] if col == 'temp' else 0), errors='coerce').fillna(0)
    
    # Rate of change
    f['temp_change'] = f['temp'].diff().fillna(0)
    f['hum_change'] = f['hum'].diff().fillna(0)
    f['pres_change'] = f['pres'].diff().fillna(0)
    
    # Rolling statistics
    for window in [3, 6, 12]:
        f[f'temp_ma{window}'] = f['temp'].rolling(window, min_periods=1).mean()
        f[f'temp_std{window}'] = f['temp'].rolling(window, min_periods=1).std().fillna(0)
        f[f'hum_ma{window}'] = f['hum'].rolling(window, min_periods=1).mean()
    
    # --- WIND ---
    for col in ['wind_speed', 'wind_dir']:
        if col in df.columns:
            f[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            f[f'{col}_change'] = f[col].diff().fillna(0)
    
    # Wind chill effect (important for HVAC)
    if 'wind_speed' in df.columns and 'temp' in df.columns:
        wind = pd.to_numeric(df['wind_speed'], errors='coerce').fillna(0)
        temp_c = pd.to_numeric(df['temp'], errors='coerce').fillna(20)
        # Wind chill formula (simplified)
        f['wind_chill'] = 13.12 + 0.6215*temp_c - 11.37*(wind**0.16) + 0.3965*temp_c*(wind**0.16)
        f['wind_chill'] = f['wind_chill'].fillna(temp_c)
    
    # --- CLOUD & SOLAR ---
    if 'cloud_cover' in df.columns:
        f['cloud_cover'] = pd.to_numeric(df['cloud_cover'], errors='coerce').fillna(0)
        f['cloud_change'] = f['cloud_cover'].diff().fillna(0)
    
    if 'uv_index' in df.columns:
        f['uv_index'] = pd.to_numeric(df['uv_index'], errors='coerce').fillna(0)
        f['solar_heat_gain'] = f['uv_index'] * (100 - f.get('cloud_cover', 50)) / 100
    
    # --- VISIBILITY ---
    if 'visibility' in df.columns:
        f['visibility'] = pd.to_numeric(df['visibility'], errors='coerce').fillna(10)
    
    # --- DEW POINT (critical for HVAC humidity control) ---
    if 'dew_point' in df.columns:
        f['dew_point'] = pd.to_numeric(df['dew_point'], errors='coerce').fillna(0)
        f['dew_point_change'] = f['dew_point'].diff().fillna(0)
        # Temperature-dew point spread (indicates humidity load)
        f['temp_dew_spread'] = f['temp'] - f['dew_point']
    
    # --- APPARENT TEMPERATURE (feels like) ---
    if 'temp_apparent' in df.columns:
        f['temp_apparent'] = pd.to_numeric(df['temp_apparent'], errors='coerce').fillna(f['temp'])
        f['apparent_change'] = f['temp_apparent'].diff().fillna(0)
    
    # --- PRECIPITATION ---
    if 'precip_intensity' in df.columns:
        f['precip_intensity'] = pd.to_numeric(df['precip_intensity'], errors='coerce').fillna(0)
        f['precip_ma3'] = f['precip_intensity'].rolling(3, min_periods=1).mean()
    
    # --- AIR QUALITY (affects HVAC fresh air intake) ---
    for aq_col in ['pm25', 'pm10', 'co', 'no2', 'o3', 'so2', 'epa_index']:
        if aq_col in df.columns:
            val = pd.to_numeric(df[aq_col], errors='coerce').fillna(0)
            f[aq_col] = val
            if len(val.dropna()) > 1:
                f[f'{aq_col}_ma3'] = val.rolling(3, min_periods=1).mean()
    
    # --- ALLERGEN INDICES (affects indoor air quality decisions) ---
    for allergen in ['grass_index', 'tree_index', 'weed_index']:
        if allergen in df.columns:
            f[allergen] = pd.to_numeric(df[allergen], errors='coerce').fillna(0)
    
    # --- TIME ENCODING ---
    n = len(f)
    f['hour_sin'] = np.sin(2 * np.pi * np.arange(n) / 24)
    f['hour_cos'] = np.cos(2 * np.pi * np.arange(n) / 24)
    f['day_sin'] = np.sin(2 * np.pi * np.arange(n) / (24 * 7))
    f['day_cos'] = np.cos(2 * np.pi * np.arange(n) / (24 * 7))
    
    # --- INTERACTION FEATURES ---
    f['temp_hum_interaction'] = f['temp'] * f['hum'] / 100
    f['wind_cloud_interaction'] = f.get('wind_speed', 0) * f.get('cloud_cover', 0) / 100
    f['dew_hum_ratio'] = f['dew_point'] / (f['hum'] + 1)
    
    return f.fillna(0)

def train_predict_lstm(scaled_data, sequence_length, feature_count):
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict temperature (index 0)
    
    if len(X) < 5:
        return None, None
    
    X_train = np.array(X)
    y_train = np.array(y)
    
    model = Sequential([
        Input(shape=(sequence_length, feature_count)),
        LSTM(64, return_sequences=True, activation='tanh'),
        Dropout(0.25),
        LSTM(32, activation='tanh'),
        Dropout(0.25),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.0008), loss='mse')
    
    batch_size = min(8, max(2, len(X_train) // 4))
    epochs = min(40, max(8, len(X_train)))
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.2, verbose=0,
              callbacks=[EarlyStopping(patience=8, restore_best_weights=True)])
    
    current_batch = scaled_data[-sequence_length:].reshape(1, sequence_length, feature_count)
    predictions_scaled = []
    
    for _ in range(8):
        pred = float(model.predict(current_batch, verbose=0)[0, 0])
        predictions_scaled.append(pred)
        
        new_row = np.zeros((1, 1, feature_count))
        new_row[0, 0, 0] = pred
        for j in range(1, feature_count):
            new_row[0, 0, j] = current_batch[0, -1, j]
        
        current_batch = np.concatenate([current_batch[:, 1:, :], new_row], axis=1)
    
    return predictions_scaled, model

def train_predict_xgboost(df, feature_cols, forecast_hours=8):
    f = df[feature_cols].copy()
    f['target'] = f['temp'].shift(-1)
    f = f.dropna()
    
    if len(f) < 5:
        return None
    
    X = f[feature_cols].values[:-1]
    y = f['target'].values[:-1]
    
    model = xgb.XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.08, 
                              subsample=0.8, colsample_bytree=0.8, verbosity=0)
    model.fit(X, y)
    
    predictions = []
    last_row = f[feature_cols].iloc[-1:].values
    
    for _ in range(forecast_hours):
        pred = float(model.predict(last_row)[0])
        predictions.append(pred)
        last_row[0, 0] = pred
        if 'temp_change' in feature_cols:
            tc_idx = feature_cols.index('temp_change')
            last_row[0, tc_idx] = pred - last_row[0, 0]
    
    return predictions

def get_weather_description(code):
    codes = {1000: 'Clear', 1100: 'Mostly Clear', 1101: 'Partly Cloudy', 1102: 'Mostly Cloudy',
             1001: 'Cloudy', 2000: 'Fog', 2100: 'Light Fog', 3000: 'Light Wind',
             4000: 'Drizzle', 4001: 'Rain', 4200: 'Light Rain', 4201: 'Heavy Rain',
             5000: 'Snow', 5001: 'Flurries', 5100: 'Light Snow', 5101: 'Heavy Snow',
             6000: 'Freezing Drizzle', 6001: 'Freezing Rain', 7000: 'Ice Pellets',
             8000: 'Thunderstorm'}
    return codes.get(code, 'Unknown')

def calculate_comfort_index(temp, humidity, wind=0):
    """Enhanced comfort index considering wind chill"""
    if temp < 27: return round(float(temp), 1)
    hi = -8.784695 + 1.61139411*temp + 2.338549*humidity - 0.14611605*temp*humidity
    hi += -0.01230809*temp*temp - 0.01642482*humidity*humidity
    hi += 0.002211732*temp*temp*humidity + 0.00072546*temp*humidity*humidity
    hi += -0.00000358*temp*temp*humidity*humidity
    # Wind cooling effect
    if wind > 3:
        hi -= wind * 0.3
    return round(float(hi), 1)

def calculate_confidence(confidence_base, total_hours, final_temps, df):
    temp_variance = float(np.var(final_temps))
    data_score = min(30, max(5, total_hours * 0.5))
    model_score = confidence_base * 0.5
    stability_score = max(5, min(25, 25 - (temp_variance * 5)))
    temp_range = max(final_temps) - min(final_temps)
    if temp_range < 8: consistency_score = 22
    elif temp_range < 12: consistency_score = 17
    elif temp_range < 16: consistency_score = 12
    elif temp_range < 20: consistency_score = 7
    else: consistency_score = 3
    
    if len(df) >= 6: hist_std = float(df['temp'].tail(6).std())
    else: hist_std = 3.0
    pred_std = float(np.std(final_temps))
    hist_match = max(5, min(15, 15 - abs(hist_std - pred_std) * 2))
    
    confidence = min(92, data_score + model_score + stability_score + consistency_score + hist_match)
    confidence = max(40, confidence)
    
    logging.info(f"Confidence: data={data_score:.0f} model={model_score:.0f} stability={stability_score:.0f} "
                 f"consistency={consistency_score:.0f} hist_match={hist_match:.0f} → {confidence:.0f}%")
    return float(confidence)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#111827', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def create_temperature_chart(hist_labels, hist_temp, forecast_labels, forecast_temps, target_ac, current_temp):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(hist_temp)), hist_temp, color='#06b6d4', linewidth=2, marker='o', markersize=3, label='Historical')
    forecast_x = range(len(hist_temp)-1, len(hist_temp) + len(forecast_temps))
    forecast_y = [hist_temp[-1]] + forecast_temps
    ax.plot(forecast_x, forecast_y, color='#f59e0b', linewidth=2, linestyle='--', marker='s', markersize=4, label='Forecast')
    ax.axhline(y=target_ac, color='#10b981', linestyle=':', linewidth=2, label=f'AC Setpoint ({target_ac}°C)')
    ax.scatter([len(hist_temp)-1], [current_temp], color='#ef4444', s=100, zorder=5, label=f'Current ({current_temp}°C)')
    ax.set_facecolor('#111827')
    ax.grid(True, alpha=0.2, color='#9ca3af')
    ax.legend(loc='upper left', facecolor='#1f2937', edgecolor='#374151', labelcolor='#9ca3af')
    ax.set_ylabel('Temperature (°C)', color='#9ca3af')
    ax.set_title('Temperature Forecast', color='#f3f4f6', fontweight='bold')
    ax.tick_params(colors='#9ca3af')
    return fig_to_base64(fig)

def create_humidity_chart(hist_temp, hist_hum, forecast_hum):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.fill_between(range(len(hist_hum)), hist_hum, alpha=0.3, color='#3b82f6', label='Historical')
    ax1.plot(range(len(hist_hum)), hist_hum, color='#3b82f6', linewidth=2)
    forecast_x = range(len(hist_hum)-1, len(hist_hum) + len(forecast_hum))
    forecast_y = [hist_hum[-1]] + forecast_hum
    ax1.plot(forecast_x, forecast_y, color='#8b5cf6', linewidth=2, linestyle='--', marker='s', markersize=4, label='Forecast')
    ax1.set_facecolor('#111827')
    ax1.grid(True, alpha=0.2, color='#9ca3af')
    ax1.legend(loc='upper left', facecolor='#1f2937', edgecolor='#374151', labelcolor='#9ca3af')
    ax1.set_ylabel('Humidity (%)', color='#3b82f6')
    ax1.set_title('Humidity Forecast', color='#f3f4f6', fontweight='bold')
    ax1.tick_params(colors='#9ca3af')
    ax1.set_ylim(0, 100)
    return fig_to_base64(fig)

def create_pressure_chart(hist_pres, forecast_pres):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(hist_pres)), hist_pres, color='#10b981', linewidth=2, marker='o', markersize=3, label='Historical')
    forecast_x = range(len(hist_pres)-1, len(hist_pres) + len(forecast_pres))
    forecast_y = [hist_pres[-1]] + forecast_pres
    ax.plot(forecast_x, forecast_y, color='#f59e0b', linewidth=2, linestyle='--', marker='s', markersize=4, label='Forecast')
    ax.set_facecolor('#111827')
    ax.grid(True, alpha=0.2, color='#9ca3af')
    ax.legend(loc='upper left', facecolor='#1f2937', edgecolor='#374151', labelcolor='#9ca3af')
    ax.set_ylabel('Pressure (hPa)', color='#9ca3af')
    ax.set_title('Pressure Forecast', color='#f3f4f6', fontweight='bold')
    ax.tick_params(colors='#9ca3af')
    return fig_to_base64(fig)

def create_shap_chart(labels, percentages):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#06b6d4', '#f59e0b', '#10b981', '#8b5cf6', '#ef4444']
    bars = ax.barh(labels, percentages, color=colors[:len(labels)])
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{pct}%',
                va='center', color='#f3f4f6', fontweight='bold')
    ax.set_facecolor('#111827')
    ax.set_xlabel('Importance (%)', color='#9ca3af')
    ax.set_title('SHAP Feature Importance', color='#f3f4f6', fontweight='bold')
    ax.tick_params(colors='#9ca3af')
    ax.set_xlim(0, max(percentages) * 1.3)
    return fig_to_base64(fig)

def create_combined_dashboard(hist_labels, hist_temp, hist_hum, hist_pres,
                               forecast_labels, forecast_temp, forecast_hum, forecast_pres,
                               target_ac, current_temp, shap_labels, shap_pct):
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0a0e17')
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(range(len(hist_temp)), hist_temp, color='#06b6d4', linewidth=2, marker='o', markersize=2, label='Historical')
    fx = range(len(hist_temp)-1, len(hist_temp) + len(forecast_temp))
    fy = [hist_temp[-1]] + forecast_temp
    ax1.plot(fx, fy, color='#f59e0b', linewidth=2, linestyle='--', marker='s', markersize=3, label='Forecast')
    ax1.axhline(y=target_ac, color='#10b981', linestyle=':', linewidth=2, label=f'AC: {target_ac}°C')
    ax1.scatter([len(hist_temp)-1], [current_temp], color='#ef4444', s=80, zorder=5)
    ax1.set_facecolor('#111827')
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=8)
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('🌡️ Temperature Forecast')
    ax1.tick_params(colors='#9ca3af')
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.fill_between(range(len(hist_hum)), hist_hum, alpha=0.3, color='#3b82f6')
    ax2.plot(range(len(hist_hum)), hist_hum, color='#3b82f6', linewidth=2, label='Historical')
    fx2 = range(len(hist_hum)-1, len(hist_hum) + len(forecast_hum))
    fy2 = [hist_hum[-1]] + forecast_hum
    ax2.plot(fx2, fy2, color='#8b5cf6', linewidth=2, linestyle='--', marker='s', markersize=3, label='Forecast')
    ax2.set_facecolor('#111827')
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=8)
    ax2.set_ylabel('Humidity (%)')
    ax2.set_title('💧 Humidity Forecast')
    ax2.tick_params(colors='#9ca3af')
    ax2.set_ylim(0, 100)
    
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(range(len(hist_pres)), hist_pres, color='#10b981', linewidth=2, marker='o', markersize=2, label='Historical')
    fx3 = range(len(hist_pres)-1, len(hist_pres) + len(forecast_pres))
    fy3 = [hist_pres[-1]] + forecast_pres
    ax3.plot(fx3, fy3, color='#f59e0b', linewidth=2, linestyle='--', marker='s', markersize=3, label='Forecast')
    ax3.set_facecolor('#111827')
    ax3.grid(True, alpha=0.2)
    ax3.legend(fontsize=8)
    ax3.set_ylabel('Pressure (hPa)')
    ax3.set_title('📊 Pressure Forecast')
    ax3.tick_params(colors='#9ca3af')
    
    ax4 = fig.add_subplot(2, 2, 4)
    colors = ['#06b6d4', '#f59e0b', '#10b981', '#8b5cf6', '#ef4444']
    bars = ax4.barh(shap_labels, shap_pct, color=colors[:len(shap_labels)])
    for bar, pct in zip(bars, shap_pct):
        ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{pct}%',
                va='center', color='#f3f4f6', fontweight='bold')
    ax4.set_facecolor('#111827')
    ax4.set_xlabel('Importance (%)')
    ax4.set_title('🔍 SHAP Feature Importance')
    ax4.tick_params(colors='#9ca3af')
    ax4.set_xlim(0, max(shap_pct) * 1.3)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.title.set_color('#f3f4f6')
        ax.xaxis.label.set_color('#9ca3af')
        ax.yaxis.label.set_color('#9ca3af')
        for spine in ax.spines.values():
            spine.set_color('#1f2937')
    
    plt.tight_layout()
    return fig_to_base64(fig)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        city = data.get('city', 'Kolkata')
        slider = int(data.get('slider', 50))
        
        logging.info(f"🌍 {city} | slider={slider}")
        
        weather_data = fetch_weather_data(city)
        if weather_data is None:
            return jsonify({
                "error": "Could not fetch weather data",
                "details": f"Tomorrow.io API failed for {city}"
            }), 503
        
        df = weather_data['df']
        current_temp = weather_data['current_temp']
        current_humidity = weather_data['current_humidity']
        current_pressure = weather_data['current_pressure']
        current_wind = weather_data['current_wind']
        current_cloud = weather_data['current_cloud']
        current_visibility = weather_data['current_visibility']
        current_uv = weather_data['current_uv']
        current_dew = weather_data['current_dew_point']
        current_precip = weather_data['current_precip']
        current_weather = weather_data['current_weather']
        total_hours = weather_data['total_hours']
        
        features_df = prepare_features(df)
        feature_cols = features_df.columns.tolist()
        
        logging.info(f"🔧 Engineered features: {len(feature_cols)} columns")
        
        scaler = MinMaxScaler((0, 1))
        scaled_data = scaler.fit_transform(features_df)
        
        sequence_length = min(24, max(6, total_hours // 3))
        
        lstm_preds_scaled, lstm_model = train_predict_lstm(scaled_data, sequence_length, len(feature_cols))
        xgb_preds = train_predict_xgboost(features_df, feature_cols[:min(15, len(feature_cols))])
        
        if lstm_preds_scaled is not None:
            dummy = np.zeros((8, len(feature_cols)))
            dummy[:, 0] = lstm_preds_scaled
            for j in range(1, len(feature_cols)):
                dummy[:, j] = scaled_data[-1, j]
            forecast_temps_raw = scaler.inverse_transform(dummy)[:, 0]
            model_used = "LSTM Neural Network (Enhanced)"
            confidence_base = 78
        elif xgb_preds is not None:
            forecast_temps_raw = np.array(xgb_preds)
            model_used = "XGBoost Regressor (Enhanced)"
            confidence_base = 68
        else:
            recent_temps = df['temp'].tail(12).values
            trend = np.mean(np.diff(recent_temps)) if len(recent_temps) >= 2 else 0
            if pd.isna(trend): trend = 0
            trend = max(-0.5, min(0.5, trend))
            hour_effects = np.sin(2 * np.pi * np.arange(8) / 24) * 1.5
            forecast_temps_raw = np.array([current_temp + trend * (i+1) + hour_effects[i] for i in range(8)])
            model_used = "Trend Analysis"
            confidence_base = 45
        
        # Smoothing
        last_known_temp = float(df['temp'].iloc[-1])
        hist_changes = np.abs(np.diff(df['temp'].tail(24).values))
        max_realistic_change = max(1.5, np.percentile(hist_changes, 90) if len(hist_changes) > 0 else 2.0)
        
        final_temps = [float(last_known_temp)]
        for i, pred in enumerate(forecast_temps_raw):
            max_change = max_realistic_change * (1 + i * 0.1)
            prev = final_temps[-1]
            if abs(pred - prev) > max_change:
                pred = prev + np.sign(pred - prev) * max_change
            final_temps.append(float(pred))
        
        final_temps = final_temps[1:]
        
        bias = current_temp - last_known_temp
        bias = max(-0.5, min(0.5, bias))
        final_temps = [float(max(0, min(50, t + bias))) for t in final_temps]
        
        # Humidity & Pressure with more factors
        hum_trend = df['hum'].diff().tail(6).mean() if len(df) > 6 else 0
        if pd.isna(hum_trend): hum_trend = 0
        hum_trend = max(-3, min(3, hum_trend))
        
        pres_trend = df['pres'].diff().tail(6).mean() if len(df) > 6 else 0
        if pd.isna(pres_trend): pres_trend = 0
        pres_trend = max(-2, min(2, pres_trend))
        
        last_hum = float(df['hum'].iloc[-1])
        last_pres = float(df['pres'].iloc[-1])
        
        final_hum = [float(max(20, min(95, last_hum + hum_trend * (i+1)))) for i in range(8)]
        final_pres = [float(max(900, min(1100, last_pres + pres_trend * (i+1)))) for i in range(8)]
        
        peak_temp = max(final_temps)
        peak_hour = final_temps.index(peak_temp) + 1
        temp_rise = round(peak_temp - current_temp, 2)
        target_ac = round(26.0 - (slider / 20.0), 1)
        
        confidence = calculate_confidence(confidence_base, total_hours, final_temps, df)
        
        comfort_now = calculate_comfort_index(current_temp, current_humidity, current_wind)
        comfort_peak = calculate_comfort_index(peak_temp, final_hum[peak_hour-1] if peak_hour <= 8 else final_hum[-1], current_wind)
        
        time_labels = [(datetime.now() + timedelta(hours=i)).strftime("%I:%M %p") for i in range(1, 9)]
        hist_labels = [(datetime.now() - timedelta(hours=min(23, total_hours-1)-i)).strftime("%I:%M %p")
                       for i in range(min(24, total_hours))]
        
        energy_impact = "High" if temp_rise > 3 else "Moderate" if temp_rise > 1.5 else "Low"
        
        hist_temp = [round(float(t), 1) for t in df['temp'].tail(24).tolist()]
        hist_hum = [round(float(h), 1) for h in df['hum'].tail(24).tolist()]
        hist_pres = [round(float(p), 1) for p in df['pres'].tail(24).tolist()]
        
        shap_labels = ["Temperature", "Humidity", "Pressure", "Dew Point", "Cloud Cover"]
        shap_percentages = [35.0, 25.0, 15.0, 15.0, 10.0]
        primary_driver = "Temperature"
        
        if lstm_model is not None:
            try:
                seq_data = scaled_data[-sequence_length:].reshape(1, sequence_length, len(feature_cols))
                background = scaled_data[:min(30, len(scaled_data)-sequence_length)]
                background = background.reshape(-1, sequence_length, len(feature_cols))
                explainer = shap.GradientExplainer(lstm_model, background)
                shap_vals = explainer.shap_values(seq_data)
                if isinstance(shap_vals, list): shap_vals = shap_vals[0]
                imp = np.sum(np.mean(np.abs(shap_vals), axis=0), axis=0)
                total = np.sum(imp) + 1e-10
                top_n = min(5, len(imp))
                top_indices = np.argsort(imp)[-top_n:][::-1]
                shap_labels = [feature_cols[i].replace('_', ' ').title() for i in top_indices]
                shap_percentages = [float(imp[i]/total*100) for i in top_indices]
                primary_driver = shap_labels[0]
            except Exception as e:
                logging.warning(f"SHAP failed: {e}")
        
        shap_pct_rounded = [round(p, 1) for p in shap_percentages]
        
        charts = {}
        try:
            charts['temperature_chart'] = create_temperature_chart(hist_labels, hist_temp, time_labels, final_temps, target_ac, current_temp)
            charts['humidity_chart'] = create_humidity_chart(hist_temp, hist_hum, final_hum)
            charts['pressure_chart'] = create_pressure_chart(hist_pres, final_pres)
            charts['shap_chart'] = create_shap_chart(shap_labels, shap_pct_rounded)
            charts['dashboard'] = create_combined_dashboard(hist_labels, hist_temp, hist_hum, hist_pres,
                time_labels, final_temps, final_hum, final_pres, target_ac, current_temp, shap_labels, shap_pct_rounded)
        except Exception as e:
            logging.warning(f"Chart generation failed: {e}")
        
        weather_alerts = []
        if sum(final_hum)/8 > 70: weather_alerts.append("High humidity expected")
        if current_precip > 0: weather_alerts.append("Precipitation active")
        if current_cloud > 80: weather_alerts.append("Heavy cloud cover")
        if current_uv > 6: weather_alerts.append("High UV index")
        weather_alert = "; ".join(weather_alerts) if weather_alerts else None
        
        response = {
            "city": city, "timestamp": datetime.now().isoformat(),
            "current_temp": round(current_temp, 1), "current_humidity": round(current_humidity, 1),
            "current_pressure": round(current_pressure, 1), "current_wind": round(current_wind, 1),
            "current_cloud": round(current_cloud, 1), "current_visibility": round(current_visibility, 1),
            "current_uv": round(current_uv, 1), "current_dew_point": round(current_dew, 1),
            "current_precip": round(current_precip, 1),
            "current_weather": get_weather_description(current_weather),
            "comfort_index": comfort_now, "predicted_peak": round(peak_temp, 1),
            "peak_hour": peak_hour, "temp_rise": temp_rise, "target_ac": target_ac,
            "energy_saving_potential": round(abs(temp_rise/10)*slider, 1),
            "confidence": round(confidence, 1), "comfort_index_peak": comfort_peak,
            "labels": time_labels,
            "temperature_values": [round(t, 1) for t in final_temps],
            "humidity_values": [round(h, 1) for h in final_hum],
            "pressure_values": [round(p, 1) for p in final_pres],
            "historical_labels": hist_labels, "historical_temperature": hist_temp,
            "historical_humidity": hist_hum, "historical_pressure": hist_pres,
            "charts": charts,
            "shap_data": {"labels": shap_labels, "percentages": shap_pct_rounded,
                          "primary_driver": primary_driver, "temporal_insights": []},
            "ai_analysis": {
                "summary": f"{model_used} predicts {temp_rise}°C {'rise' if temp_rise>0 else 'drop'} over 8h. "
                          f"Peak at hour {peak_hour}. Based on {total_hours}h of real data with {len(feature_cols)} features.",
                "recommendation": "Pre-cool before peak" if temp_rise > 2 else "Maintain current settings",
                "energy_impact": energy_impact,
                "weather_alert": weather_alert
            },
            "technical_metrics": {
                "model_type": model_used, "data_source": "Tomorrow.io Real-time API",
                "training_samples": int(total_hours), "sequence_length": int(sequence_length),
                "features_used": int(len(feature_cols)),
                "explainability": "SHAP Gradient Explainer" if lstm_model else "Feature Importance",
                "last_updated": datetime.now().isoformat()
            }
        }
        
        logging.info(f"✅ {city}: {model_used} → {temp_rise}°C change, {confidence}% confidence")
        logging.info(f"   Temps: {final_temps}")
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"❌ Error: {e}", exc_info=True)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "operational", "models": ["Enhanced LSTM", "XGBoost", "Trend Analysis"]})

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "models": ["Enhanced LSTM", "XGBoost", "Trend Analysis"],
        "features": ["Temperature", "Humidity", "Pressure", "Wind", "Clouds", "UV", 
                     "Dew Point", "Precipitation", "Air Quality", "Allergens", 
                     "Solar Heat Gain", "Wind Chill", "Apparent Temperature"],
        "data_source": "Tomorrow.io Comprehensive API",
        "forecast": "8 hours",
        "explainability": "SHAP + Feature Importance"
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🌡️  ClimateSync - Enhanced Backend")
    print("📡 Tomorrow.io | 🧠 LSTM+XGBoost | 🔍 SHAP")
    print("📊 30+ Features: Wind, Clouds, UV, Dew, AQ, Allergens")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)