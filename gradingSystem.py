import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.integrate import cumtrapz
import matlab.engine
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import os

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
    accelerations = np.stack(positions, axis=-1)
    times = np.asarray(times, dtype='float64')
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("Time values should be strictly increasing.")
    jerks = np.diff(accelerations, axis=0) / dt[:, None]
    jerk_norms = np.linalg.norm(jerks, axis=1)
    L2_norm = np.mean(jerk_norms)
    return L2_norm


def run_matlab_scripts(datafile_path):
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Change to the directory containing your MATLAB script
    eng.cd('/Users/adelawu/Desktop/MRes/OnTrack_Rehab/ArmTroi-main/data')  # Replace with the actual path

    try:
        # Set the datafile variable in the MATLAB workspace
        eng.workspace['datafile'] = datafile_path

        # Run the MATLAB script 'new.m'
        eng.eval('new', nargout=0)

        # Change to the directory for the next MATLAB script
        eng.cd('/Users/adelawu/Desktop/MRes/OnTrack_Rehab/ArmTroi-main/src/stateEstimation')

        # Run the MATLAB script 'stateEstimation_overall.m'
        wrist, elbow = eng.stateEstimation_overall(nargout=2)

        # Convert MATLAB arrays to Python arrays
        wrist_py = [list(row) for row in wrist]
        elbow_py = [list(row) for row in elbow]
        return wrist_py, elbow_py
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop MATLAB engine
        eng.quit()


def process_file(file_path):
    file_data = pd.read_csv(file_path)
    if len(file_data) >= 350:
        file_data = file_data.tail(350).reset_index(drop=True)
    # _, side, _, _, _ = file_path[:-2].split('-')
    # if side == 'R':
    #     file_data['accelerationX'] = -file_data['accelerationX']
    #     # file_data['accelerationZ'] = -file_data['accelerationZ']
    #     file_data['rotationRateX'] = -file_data['rotationRateX']
    #     file_data['rotationRateZ'] = -file_data['rotationRateZ']
    #     # Adjust gravity sensors
    #     file_data['gravityX'] = -file_data['gravityX']
    #     # file_data['gravityZ'] = -file_data['gravityZ']
    #     # gravityY remains unchanged
    #
    #     # Adjust attitude sensors
    #     file_data['attitudeRoll'] = -file_data['attitudeRoll']
    #     file_data['attitudeYaw'] = -file_data['attitudeYaw']

    file_data["Timestamp"] = pd.to_datetime(file_data["Timestamp"], unit='s')
    file_data["Date"] = file_data["Timestamp"].dt.date

    cutoff = 6  # Cutoff frequency, Hz
    fs = 20  # Sampling frequency
    file_data['accelerationX'] = butter_lowpass_filter(file_data['accelerationX'], cutoff, fs)
    file_data['accelerationY'] = butter_lowpass_filter(file_data['accelerationY'], cutoff, fs)
    file_data['accelerationZ'] = butter_lowpass_filter(file_data['accelerationZ'], cutoff, fs)

    times = np.array(pd.to_datetime(file_data['Timestamp']), dtype='datetime64[ns]')
    times_in_seconds = (times - times[0]).astype('float64') / 1e9

    a_x = file_data['accelerationX'].values
    a_y = file_data['accelerationY'].values
    a_z = file_data['accelerationZ'].values
    ##Metrix1: Jerk
    jerk_norm = calculate_jerk_norm((a_x, a_y, a_z), times_in_seconds)
    ##Metrix2: ROM
    rom = np.max(np.abs(np.degrees(file_data['attitudeRoll'])))
    # rom = np.ptp((a_x, a_y, a_z), axis=0)  # Calculate the ROM for each dimension
    ##Metric3: Trajectory
    wrist, elbow = run_matlab_scripts(file_path)

    return jerk_norm, rom, wrist, elbow


def scale_trajectories(trajectory):
    norm = np.linalg.norm(trajectory)
    return trajectory / norm if norm != 0 else trajectory


def dtw_rms_distance_scaled(R, O):
    # Convert to numpy arrays
    R = np.array(R)
    O = np.array(O)

    # Compute the DTW distance and the optimal path
    distance, path = fastdtw(R, O, dist=euclidean)

    # Align the trajectories based on the optimal path
    aligned_R = np.array([R[idx] for idx, _ in path])
    aligned_O = np.array([O[idx] for _, idx in path])

    # Compute the squared differences
    squared_diffs = np.sum(np.linalg.norm(aligned_R - aligned_O, axis=1) ** 2)

    # Compute the RMS distance
    rms_dist = np.sqrt(squared_diffs / len(path))

    return rms_dist


def map_combined_metric_to_score(elbow_rms_results_scaled, jerk_diff, class_possibility,
                                 wrist_rms_results_scaled, ROM_diff,
                                 min_rms=0, max_rms=1,
                                 weight_rms=0.2, weight_rom=0.1, weight_jerk=0.1, weight_class=0.3,
                                 weight_wrist_rms=0.2):
    """
    Combine the scaled RMS distance, ROM difference, jerk difference, class possibility,
    wrist RMS distance into a single score.
    Normalize and combine these metrics using the specified weights.
    The combined metric is then mapped to a score in the range [0, 10].
    """
    if not (0 <= elbow_rms_results_scaled <= max_rms):
        raise ValueError(f"Scaled RMS distance should be in the range [0, {max_rms}].")
    if not (0 <= class_possibility <= 1):
        raise ValueError("Class possibility should be in the range [0, 1].")
    if not (0 <= wrist_rms_results_scaled <= max_rms):
        raise ValueError(f"Wrist scaled RMS distance should be in the range [0, {max_rms}].")
    if ROM_diff < 0:
        raise ValueError("ROM difference should be non-negative.")
    if jerk_diff < 0:
        raise ValueError("Jerk difference should be non-negative.")

    # Normalize scaled RMS distance
    normalized_rms = elbow_rms_results_scaled / (max_rms - min_rms)
    normalized_wrist_rms = wrist_rms_results_scaled / (max_rms - min_rms)

    # Invert the normalized RMS distances (higher RMS should lower the score)
    inverted_rms = 1 - elbow_rms_results_scaled
    inverted_wrist_rms = 1 - wrist_rms_results_scaled

    # Invert ROM_diff and jerk_diff since smaller values are better
    inverted_rom_diff = 1 - ROM_diff
    inverted_jerk_diff = 1 - jerk_diff

    # Combine the normalized/inverted values using the given weights
    combined_metric = (
            (weight_rms * inverted_rms) +
            (weight_rom * inverted_rom_diff) +
            (weight_jerk * inverted_jerk_diff) +
            (weight_class * class_possibility) +
            (weight_wrist_rms * inverted_wrist_rms)
    )

    # Map the combined metric to a score in the range [0, 10]
    score = combined_metric * 10

    # Print the values of the metrics for the report
    print("Metrics Report:")
    print(f"Elbow RMS Results Scaled: {inverted_rms}")
    print(f"Jerk Difference: {inverted_jerk_diff}")
    print(f"Class Possibility: {class_possibility}")
    print(f"Wrist RMS Results Scaled: {inverted_wrist_rms}")
    print(f"ROM Difference: {inverted_rom_diff}")
    print(f"Combined Metric: {combined_metric}")
    print(f"Score: {score}")

    return score


def calculate_score(class_possibility, new_file_path="/Users/adelawu/PycharmProjects/OnTrack/result.csv", benchmark_file_path="GERF-L-D001-M6-S0043.csv"):
    new_jerk, new_rom, new_wrist, new_elbow = process_file(new_file_path)
    benchmark_jerk, benchmark_rom, benchmark_wrist, benchmark_elbow = process_file(benchmark_file_path)

    jerk_diff = np.abs(new_jerk - benchmark_jerk)
    rom_diff = np.abs(new_rom - benchmark_rom)
    wrist_distance = dtw_rms_distance_scaled(new_wrist, benchmark_wrist)
    elbow_distance = dtw_rms_distance_scaled(new_elbow, benchmark_elbow)

    # print(f"New file ROM value: {new_rom}")
    # print(f"Benchmark ROM value: {benchmark_rom}")
    # print(f"Jerk difference: {jerk_diff}")
    # print(f"ROM difference: {rom_diff}")
    # print(f"Wrist Distance: {wrist_distance}")
    # print(f"Elbow Distance: {elbow_distance}")

    # Call the function
    score = map_combined_metric_to_score(elbow_distance, jerk_diff, class_possibility,
                                         wrist_distance, rom_diff)

    return score


# if __name__ == "__main__":
#     class_possibility = 0.6  # Example class possibility value
#     score = calculate_score(class_possibility)
#     print(f"Final Score: {score}")
