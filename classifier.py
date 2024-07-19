import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import joblib

# Lowpass filter implementation
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

# Preprocessing function
def load_and_preprocess(data):
    data = data.dropna()
    data = data.loc[:, (data != data.iloc[0]).any()]
    for col in data.columns[:-1]:
        data[col] = butter_lowpass_filter(data[col], cutoff=3.0, fs=50.0, order=2)
    scaler = StandardScaler()
    data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])
    return data

# Spectral feature extraction
def calculate_spectral_features(segment):
    fft_vals = fft(segment, axis=0)
    fft_freqs = np.fft.fftfreq(len(segment), d=1.0)
    mag_fft = np.abs(fft_vals)
    if mag_fft.ndim > 1:
        fft_freqs = fft_freqs[:, np.newaxis]
    centroid = np.sum(fft_freqs * mag_fft, axis=0) / np.sum(mag_fft, axis=0)
    bandwidth = np.sqrt(np.sum(((fft_freqs - centroid)**2) * mag_fft, axis=0) / np.sum(mag_fft, axis=0))
    flatness = 10 ** (np.mean(np.log10(mag_fft + 1e-10), axis=0) - np.log10(np.mean(mag_fft, axis=0) + 1e-10))
    variance = np.sum(((fft_freqs - centroid)**2) * mag_fft, axis=0) / np.sum(mag_fft, axis=0)
    return np.array([centroid]), np.array([bandwidth]), np.array([flatness]), np.array([variance])

# Feature extraction function
def extract_features(data, window_size, step_size, exclude_columns=None):
    if exclude_columns is not None:
        data = data.drop(columns=exclude_columns)
    global_features = []
    for column in data.columns:
        column_data = data[column].values
        mean_features = np.array([column_data.mean()])
        max_features = np.array([column_data.max()])
        min_features = np.array([column_data.min()])
        std_features = np.array([column_data.std()])
        skew_features = np.array([skew(column_data, axis=0)])
        kurt_features = np.array([kurtosis(column_data, axis=0)])
        centroid_features, bandwidth_features, flatness_features, variance_features = calculate_spectral_features(column_data)
        combined_features = np.concatenate([
            mean_features, max_features, min_features, std_features, skew_features, kurt_features,
            centroid_features, bandwidth_features, flatness_features, variance_features
        ])
        global_features.append(combined_features)
    global_features = np.concatenate(global_features)
    return global_features

# Main script
if __name__ == "__main__":
    path_to_csv_files = 'Movement_data'
    files = os.listdir(path_to_csv_files)
    all_features = []
    all_labels = []

    for file in files:
        _, _, _, movement = file[:-4].split('-')
        label = int(movement[1]) - 1
        data = pd.read_csv(os.path.join(path_to_csv_files, file))
        if len(data) < 100:
            continue
        df = data
        side, _, _, _ = file[:-1].split('-')
        if side == 'R':
            df['accelerationX'] = -df['accelerationX']
            df['accelerationZ'] = -df['accelerationZ']
            df['rotationRateX'] = -df['rotationRateX']
            df['rotationRateZ'] = -df['rotationRateZ']
            df['gravityX'] = -df['gravityX']
            df['gravityZ'] = -df['gravityZ']
            df['attitudeRoll'] = -df['attitudeRoll']
            df['attitudeYaw'] = -df['attitudeYaw']
        features = extract_features(df, window_size=128, step_size=64, exclude_columns=['Timestamp'])
        all_features.append(features)
        all_labels.append(label)

    X = np.array(all_features)
    y = np.array(all_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)
    joblib.dump(rf, 'classifier_model.joblib')