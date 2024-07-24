import socket
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import matplotlib.dates as mdates
import joblib
import numpy as np
import pandas as pd
from scipy.signal import lfilter, butter
from scipy.stats import skew, kurtosis
import numpy as np
from scipy.stats import skew, kurtosis
from numpy.fft import fft
from sklearn.preprocessing import StandardScaler


# Shared data structure for threads
data = pd.DataFrame(columns=['Timestamp', 'attitudeRoll', 'accelerationX', 'accelerationY', 'accelerationZ'])
data_lock = threading.Lock()
initial_angle = None

# Variables to hold previous state
previous_stage = 1
previous_roll = None
previous_velX_trend = None
velocities = []
time_seconds = []
origin_time = None

def smooth_data(data, column_name, window_size=3):
    """Smooth data using a simple moving average."""
    return data[column_name].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')

def calculate_trend(data):
    """Calculate trend based on whether the data is increasing or decreasing."""
    return data.diff().fillna(0) >= 0

def is_stable(current_angle, previous_angle, min_angle=50, change_limit=50):
    """
    Check if the angle change is stable.
    """
    return current_angle >= min_angle and abs(current_angle - previous_angle) < change_limit

def calculate_stages(current_index):
    global previous_stage, previous_roll, previous_velX_trend, velocities, time_seconds, origin_time

    color_map = {1: 'yellow', 2: 'green', 3: 'red', 4: 'purple'}

    if current_index == 0:
        return 'yellow', 1
    velocities = []
    time_seconds = []
    current_row = data.iloc[current_index]
    previous_row = data.iloc[current_index - 2]

    roll_current = np.abs(np.degrees(current_row['SmoothedRoll']))
    roll_previous = np.abs(np.degrees(previous_row['SmoothedRoll']))
    velX_current = current_row['SmoothedVelX']
    velX_previous = previous_row['SmoothedVelX']
    mag_current = current_row['velocityMagnitude']
    timestamp_current = current_row['Timestamp']

    roll_trend = roll_current >= roll_previous
    velX_trend = velX_current >= velX_previous

    new_stage = previous_stage  # Default to previous stage

    # Apply the new stability definition
    if is_stable(roll_current, roll_previous):
        if previous_stage == 1 and velX_current < -0.05:
            new_stage = 2  # Stable and velocity Y is not increasing
            if previous_stage == 1:
                # Calculate Stage 1 score at the transition to Stage 2
                score = np.abs(roll_current) / 160
                print(f"Stage 1 Score at {previous_row['Timestamp']}: {score}")
                origin_time = timestamp_current
                velocities = []
                time_seconds = []
        if previous_stage == 2:
            if roll_current >= 50 and velX_trend != previous_velX_trend and velX_current > 0:
                new_stage = 3
                if previous_stage == 2:
                    # Calculate Stage 2 score at the transition to Stage 3
                    if velocities and time_seconds:
                        velocities = np.array(velocities)
                        time_seconds = np.array(time_seconds)
                        cumulative_distance = np.trapz(np.abs(velocities), time_seconds)
                        score = cumulative_distance / 0.06
                        print(f"Stage 2 Score at {previous_row['Timestamp']}: {score}")
                    velocities = []
                    time_seconds = []
            else:
                new_stage = 2
    else:
        if roll_trend:
            new_stage = 1  # Stage 1 condition
        elif (roll_previous - roll_current >= 30):
            new_stage = 4


    previous_stage = new_stage

    # Collect velocities and timestamps for numerical integration if in Stage 2
    if new_stage == 2:
        if origin_time is not None:
            time_elapsed = (timestamp_current - origin_time).total_seconds()
            time_seconds.append(time_elapsed)
            velocities.append(mag_current)

    previous_roll = roll_current
    previous_velX_trend = velX_trend

    # Color assignment based on the current stage
    color = color_map.get(new_stage, 'gray')

    return color, new_stage

class DynamicPlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Plot in Tkinter")

        self.fig, self.ax1 = plt.subplots(figsize=(14, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Create a twinx axis for the second y-axis
        self.ax2 = self.ax1.twinx()

        # Initial plot setup for attitudeRoll
        self.line1, = self.ax1.plot([], [], color='blue', marker='o', label='Attitude Roll')
        self.ax1.set_xlabel('Timestamp')
        self.ax1.set_ylabel('Attitude Roll (degrees)', color='blue')
        self.ax1.tick_params(axis='y', labelcolor='blue')

        # Initial plot setup for velocityMagnitude
        self.scatter = self.ax2.scatter([], [], color='red', label='Velocity Magnitude')
        self.ax2.set_ylabel('Velocity Magnitude', color='red')
        self.ax2.tick_params(axis='y', labelcolor='red')

        # Add a button to start the receiver thread
        # self.start_button = ttk.Button(root, text="Start", command=self.start_receiver_thread)
        # self.start_button.pack(side=tk.BOTTOM)

        self.running = False
        self.start_receiver_thread()

    def start_receiver_thread(self):
        self.running = True
        self.receiver_thread = threading.Thread(target=self.data_receiver)
        self.receiver_thread.start()
        self.update_plot()

    def data_receiver(self):
        """Thread function to receive data from the client and update the shared data."""
        global initial_angle
        while self.running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                    server_socket.bind(('0.0.0.0', 12345))
                    server_socket.listen(1)
                    server_socket.settimeout(10)
                    print("Server is listening for incoming connections...")
                    try:
                        connection, client_address = server_socket.accept()
                    except socket.timeout:
                        continue

                    print(f"Connection from {client_address}")
                    with connection:
                        connection.settimeout(10)
                        buffer = ""
                        header = True
                        column_map = {}
                        with open('result.csv', 'w') as f:
                            while self.running:
                                try:
                                    data_chunk = connection.recv(1024).decode()
                                    if not data_chunk:
                                        break
                                    buffer += data_chunk
                                    while '\n' in buffer:
                                        line, buffer = buffer.split('\n', 1)
                                        if header:
                                            headers = line.split(',')
                                            column_map = {key: index for index, key in enumerate(headers)}
                                            header = False
                                            f.write(line + '\n')
                                        else:
                                            print(line)  # Print each received data line to the console
                                            f.write(line + '\n')
                                            parts = line.split(',')
                                            if len(parts) >= 13:  # Ensure there are enough parts in the line
                                                try:
                                                    timestamp = float(parts[0])
                                                    attitudeRoll = float(parts[10])
                                                    accelerationX = float(parts[4])
                                                    accelerationY = float(parts[5])
                                                    accelerationZ = float(parts[6])
                                                    if initial_angle is None:
                                                        initial_angle = attitudeRoll  # Set the initial angle

                                                    attitudeRoll -= initial_angle  # Reset the starting angle to zero

                                                    with data_lock:
                                                        data_row = pd.DataFrame([{
                                                            'Timestamp': pd.to_datetime(timestamp, unit='s'),
                                                            'attitudeRoll': attitudeRoll,
                                                            'accelerationX': accelerationX,
                                                            'accelerationY': accelerationY,
                                                            'accelerationZ': accelerationZ
                                                        }])
                                                        global data
                                                        data = pd.concat([data, data_row], ignore_index=True)

                                                except ValueError:
                                                    print("ValueError: Could not convert data")  # Debugging statement
                                                    continue

                                                # Process the new data point outside the lock
                                                self.process_new_data_point()
                                except socket.timeout:
                                    print("Socket timeout, attempting to continue...")
                                    break
                                except ConnectionResetError:
                                    print("Connection reset by peer, attempting to continue...")
                                    break
                    # Process the received file after connection closes
                    self.process_received_file()
            except Exception as e:
                print(f"Exception: {e}")
                continue
    def process_received_file(self):
        try:
            # Load the new data
            new_data = pd.read_csv('result.csv')

            # Process the data to get features
            features = extract_features(new_data[1:], window_size=100, step_size=50, exclude_columns=['Timestamp'])
            all_features = []
            all_features.append(features)

            # Convert features to numpy array
            new_input = np.array(all_features)

            # Load the model
            clf = joblib.load('classifier_model.joblib')

            # Make predictions
            predictions = clf.predict(new_input)

            # Print or save the predictions
            print(f"Classification result: Motion{predictions+1}")


        except Exception as e:
            print(f"Error processing received file: {e}")





    def process_new_data_point(self):
        """Process the latest data point to calculate the stage and color and update the plot."""
        with data_lock:
            if not data.empty:
                current_index = len(data) - 1

                # Remove the mean (bias) from the acceleration data
                data['accelerationXD'] = data['accelerationX'] - 0.058687863613335826
                data['accelerationYD'] = data['accelerationY'] - -0.016740100437335752
                data['accelerationZD'] = data['accelerationZ'] - 0.022834287908395628
                # data['accelerationXD'] = smooth_data(data, 'accelerationX')
                # data['accelerationYD'] = smooth_data(data, 'accelerationY')
                # data['accelerationZD'] = smooth_data(data, 'accelerationZ')

                # Integration to get velocity
                data['velocityX'] = np.cumsum(data['accelerationXD']) / 20
                data['velocityY'] = np.cumsum(data['accelerationYD']) / 20
                data['velocityZ'] = np.cumsum(data['accelerationZD']) / 20

                # Calculate the velocity magnitude
                data['velocityMagnitude'] = np.sqrt(data['velocityX'] ** 2 + data['velocityY'] ** 2 + data['velocityZ'] ** 2)

                # Smooth the data
                data['SmoothedRoll'] = smooth_data(data, 'attitudeRoll')
                data['SmoothedVelX'] = smooth_data(data, 'accelerationX')

                # Calculate the color and stage for the latest data point
                color, stage = calculate_stages(current_index)

                # Convert Timestamp to matplotlib date format for plotting
                timestamp = mdates.date2num(data.iloc[current_index]['Timestamp'])

                # Update the scatter plot with the new point
                self.ax2.scatter([timestamp], [data.iloc[current_index]['velocityMagnitude']], color=color)

                # Update the attitudeRoll line plot
                self.line1.set_data(mdates.date2num(data['Timestamp']), np.abs(np.degrees(data['attitudeRoll'])))
                self.ax1.relim()
                self.ax1.autoscale_view()

                # Redraw the canvas to show the new point
                self.canvas.draw()

    def update_plot(self):
        """Update the plot dynamically."""
        self.canvas.draw()
        if self.running:
            self.root.after(1000, self.update_plot)  # Schedule the next update

    def stop(self):
        self.running = False
        self.receiver_thread.join()


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def load_and_preprocess(data):
    # Clean data: Assume dropping NA and any constant columns as example
    data = data.dropna()
    data = data.loc[:, (data != data.iloc[0]).any()]

    # Filter data: Example with a low-pass filter
    for col in data.columns[:-1]:  # Assuming last column is not part of the filtering
        data[col] = butter_lowpass_filter(data[col], cutoff=3.0, fs=50.0, order=2)

    # Normalize data
    scaler = StandardScaler()
    data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

    return data





def calculate_spectral_features(segment):
    # Compute the FFT
    fft_vals = fft(segment, axis=0)
    fft_freqs = np.fft.fftfreq(len(segment), d=1.)  # Assuming unit sampling frequency

    # Magnitude of FFT
    mag_fft = np.abs(fft_vals)

    # Reshape fft_freqs for broadcasting if necessary
    if mag_fft.ndim > 1:
        fft_freqs = fft_freqs[:, np.newaxis]

    # Spectral centroid
    centroid = np.sum(fft_freqs * mag_fft, axis=0) / np.sum(mag_fft, axis=0)

    # Spectral bandwidth
    bandwidth = np.sqrt(np.sum(((fft_freqs - centroid)**2) * mag_fft, axis=0) / np.sum(mag_fft, axis=0))

    # Spectral flatness
    flatness = 10 ** (np.mean(np.log10(mag_fft + 1e-10), axis=0) - np.log10(np.mean(mag_fft, axis=0) + 1e-10))

    # Spectral variance
    variance = np.sum(((fft_freqs - centroid)**2) * mag_fft, axis=0) / np.sum(mag_fft, axis=0)

    return np.array([centroid]), np.array([bandwidth]), np.array([flatness]), np.array([variance])

def extract_features(data, window_size, step_size, exclude_columns=None):
    if exclude_columns is not None:
        data = data.drop(columns=exclude_columns)

    global_features = []  # Corrected initialization
    for column in data.columns:
        column_data = data[column].values
        mean_features = np.array([column_data.mean()])  # Ensure scalar values are wrapped in an array
        max_features = np.array([column_data.max()])
        min_features = np.array([column_data.min()])
        std_features = np.array([column_data.std()])
        skew_features = np.array([skew(column_data, axis=0)])
        kurt_features = np.array([kurtosis(column_data, axis=0)])

        # Frequency domain features
        centroid_features, bandwidth_features, flatness_features, variance_features = calculate_spectral_features(column_data)
        # Combine all features into one array
        combined_features = np.concatenate([
            mean_features, max_features, min_features, std_features, skew_features, kurt_features,
            centroid_features, bandwidth_features, flatness_features, variance_features
        ])
        global_features.append(combined_features)

    global_features = np.concatenate(global_features)  # Corrected to concatenate list of arrays

    return global_features






if __name__ == "__main__":
    root = tk.Tk()
    app = DynamicPlotApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop)  # Ensure the receiver thread is stopped on window close
    root.mainloop()
