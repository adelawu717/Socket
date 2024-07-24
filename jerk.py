import os
import pandas as pd
import datetime
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set print options for numpy
np.set_printoptions(precision=4)

# Butterworth lowpass filter function
def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Compute trajectory function
def compute_trajectory(ax, ay, az, dt):
    vx = cumtrapz(ax, dx=dt, initial=0)
    vy = cumtrapz(ay, dx=dt, initial=0)
    vz = cumtrapz(az, dx=dt, initial=0)
    x = cumtrapz(vx, dx=dt, initial=0)
    y = cumtrapz(vy, dx=dt, initial=0)
    z = cumtrapz(vz, dx=dt, initial=0)
    return x, y, z

# Calculate jerk norm function
def calculate_jerk_norm(positions, times):
    # Stack the acceleration arrays to form a 2D array of shape (Tn, 3)
    accelerations = np.stack(positions, axis=-1)
    times = np.asarray(times, dtype='float64')

    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("Time values should be strictly increasing.")

    jerks = np.diff(accelerations, axis=0) / dt[:, None]

    # norm of the jerk at each point
    jerk_norms = np.linalg.norm(jerks, axis=1)

    # average norm of the jerk
    L2_norm = np.mean(jerk_norms)

    return L2_norm


def main():
    root_directory = 'Standard_data/'  # Replace this with the path to your directory

    # Initialize file collections for each category
    file_collections = {f'M{i}': [] for i in range(1, 7)}

    # Walk through each folder and subfolder in the given root directory
    for subdir, dirs, files in os.walk(root_directory):
        for file in files:
            for category in file_collections.keys():
                if category in file:
                    full_file_path = os.path.join(subdir, file)
                    file_data = pd.read_csv(full_file_path)

                    if len(file_data) > 100:
                        _, side, _, _, _ = file[:-2].split('-')
                        if side == 'R':
                            file_data['accelerationX'] = -file_data['accelerationX']
                            file_data['rotationRateX'] = -file_data['rotationRateX']
                            file_data['rotationRateZ'] = -file_data['rotationRateZ']
                            file_data['gravityX'] = -file_data['gravityX']
                            file_data['attitudeRoll'] = -file_data['attitudeRoll']
                            file_data['attitudeYaw'] = -file_data['attitudeYaw']

                        file_data["Timestamp"] = pd.to_datetime(file_data["Timestamp"], unit='s')
                        file_data["Date"] = file_data["Timestamp"].dt.date

                        # Apply Butterworth low-pass filter
                        cutoff = 6  # Cutoff frequency, Hz
                        fs = 20     # Sampling frequency
                        file_data['accelerationX'] = butter_lowpass_filter(file_data['accelerationX'], cutoff, fs)
                        file_data['accelerationY'] = butter_lowpass_filter(file_data['accelerationY'], cutoff, fs)
                        file_data['accelerationZ'] = butter_lowpass_filter(file_data['accelerationZ'], cutoff, fs)

                        file_collections[category].append(file_data)

    # Initialize collections for trajectories and jerk
    O_collections = {f'M{i}': [] for i in range(1, 7)}
    jerk_collections = {f'M{i}': [] for i in range(1, 7)}

    # Process each category
    for category, files_data in file_collections.items():
        for file_data in files_data:
            # time = file_data['Timestamp'].values
            dt = 0.01

            a_x = file_data['accelerationX'].values
            a_y = file_data['accelerationY'].values
            a_z = file_data['accelerationZ'].values

            # Compute the trajectory
            x, y, z = compute_trajectory(a_x, a_y, a_z, dt)
            O_collections[category].append(np.array(np.stack((x, y, z), axis=-1)))
            times = np.array(pd.to_datetime(file_data['Timestamp']), dtype='datetime64[ns]')
            times_in_seconds = (times - times[0]).astype('float64') / 1e9
            # Calculate jerk norm
            norm_of_jerk = calculate_jerk_norm((a_x, a_y, a_z), times_in_seconds)
            jerk_collections[category].append(norm_of_jerk)

    # Now, O_collections and jerk_collections contain the processed data
    print("Trajectory and jerk calculations complete.")

    data = pd.read_csv("GERF-L-D001-M6-S0043.csv")

    # Ensure data length is capped at 350
    if len(data) >= 350:
        data = data.tail(350).reset_index(drop=True)
    ## calculate jerk for test file
    M6_jerk = jerk_collections['M6']
    file_data = data
    file_data["Timestamp"] = file_data["Timestamp"].map(
        lambda t: datetime.datetime.fromtimestamp(t) if isinstance(t, float) else t)
    file_data["Date"] = file_data["Timestamp"].map(lambda t: t.date())

    cutoff = 6  # Cutoff frequency, Hz
    fs = 20
    file_data['accelerationX'] = butter_lowpass_filter(file_data['accelerationX'], cutoff, fs, order=2)
    file_data['accelerationY'] = butter_lowpass_filter(file_data['accelerationY'], cutoff, fs, order=2)
    file_data['accelerationZ'] = butter_lowpass_filter(file_data['accelerationZ'], cutoff, fs, order=2)

    # Calculate jerk for each axis
    time = np.array(file_data['Timestamp'])
    t1 = time[0].astype('datetime64[s]').astype(int)
    t2 = time[-1].astype('datetime64[s]').astype(int)
    duration = t2 - t1  # Total time of simulation
    dt = 0.01  # Time step

    time_points = np.array(pd.to_datetime(file_data['Timestamp']), dtype='datetime64[ns]')


    a_x = np.array(file_data['accelerationX'])
    a_y = np.array(file_data['accelerationY'])
    a_z = np.array(file_data['accelerationZ'])


    # Compute the trajectory by integrating the acceleration data
    x, y, z = compute_trajectory(a_x, a_y, a_z, dt)
    R = np.array(np.stack((x, y, z), axis=-1))
    times = np.array(pd.to_datetime(file_data['Timestamp']), dtype='datetime64[ns]')
    times_in_seconds = (times - times[0]).astype('float64') / 1e9
    R_jerk = calculate_jerk_norm((a_x, a_y, a_z), times_in_seconds)
    # Step 2: Calculate the z-score of the new input
    from scipy.stats import norm
    def calculate_z_score(new_input, mean, std_dev):
        return (new_input - mean) / std_dev

    # Step 3: Use the z-score to find the probability
    def calculate_probability(z_score):
        # Using the cumulative distribution function (CDF) of the normal distribution
        probability = norm.cdf(z_score)
        return probability

    mean = np.mean(M6_jerk)
    print(f"jerk mean for M6: {mean}")
    std_dev = np.std(M6_jerk)
    z_score = calculate_z_score(R_jerk, mean, std_dev)
    jerk_probability = calculate_probability(z_score)
    print("jerk probability:", jerk_probability)

if __name__ == "__main__":
    main()