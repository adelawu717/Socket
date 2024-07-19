import socket
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import datetime as dt

# Global variable and lock for data
data = pd.DataFrame(columns=[
    'Timestamp', 'gravityX', 'gravityY', 'gravityZ',
    'accelerationX', 'accelerationY', 'accelerationZ',
    'rotationRateX', 'rotationRateY', 'rotationRateZ',
    'attitudeRoll', 'attitudePitch', 'attitudeYaw'
])
data_lock = threading.Lock()

class DynamicPlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Plot in Tkinter")

        # Create a frame for the plot
        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.axs = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Subplot 1: Gravity (XYZ)
        self.ax1 = self.axs[0]
        self.line1_x, = self.ax1.plot([], [], color='blue', label='Gravity X')
        self.line1_y, = self.ax1.plot([], [], color='green', label='Gravity Y')
        self.line1_z, = self.ax1.plot([], [], color='red', label='Gravity Z')
        self.ax1.set_ylabel('Gravity (m/s²)')
        self.ax1.legend()

        # Subplot 2: Acceleration (XYZ)
        self.ax2 = self.axs[1]
        self.line2_x, = self.ax2.plot([], [], color='blue', label='Acceleration X')
        self.line2_y, = self.ax2.plot([], [], color='green', label='Acceleration Y')
        self.line2_z, = self.ax2.plot([], [], color='red', label='Acceleration Z')
        self.ax2.set_ylabel('Acceleration (m/s²)')
        self.ax2.legend()

        # Subplot 3: Rotation Rate (XYZ)
        self.ax3 = self.axs[2]
        self.line3_x, = self.ax3.plot([], [], color='blue', label='Rotation Rate X')
        self.line3_y, = self.ax3.plot([], [], color='green', label='Rotation Rate Y')
        self.line3_z, = self.ax3.plot([], [], color='red', label='Rotation Rate Z')
        self.ax3.set_ylabel('Rotation Rate (rad/s)')
        self.ax3.legend()

        # Subplot 4: Attitude (Roll, Pitch, Yaw)
        self.ax4 = self.axs[3]
        self.line4_roll, = self.ax4.plot([], [], color='blue', label='Roll')
        self.line4_pitch, = self.ax4.plot([], [], color='green', label='Pitch')
        self.line4_yaw, = self.ax4.plot([], [], color='red', label='Yaw')
        self.ax4.set_xlabel('Timestamp')
        self.ax4.set_ylabel('Attitude (degrees)')
        self.ax4.legend()

        self.running = False

        # Start the receiver thread directly
        self.start_receiver_thread()

        # Set up the animation
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)

    def process_new_data_point(self):
        """Process the latest data point to update the plot."""
        with data_lock:
            if not data.empty:
                # Define the window length (e.g., 10 seconds)
                window_length = pd.Timedelta(seconds=10)
                current_time = data['Timestamp'].max()
                window_start_time = current_time - window_length

                # Filter the data to only include the most recent window
                window_data = data[data['Timestamp'] >= window_start_time]

                timestamps = mdates.date2num(window_data['Timestamp'])

                # Update Gravity subplot
                self.line1_x.set_data(timestamps, window_data['gravityX'])
                self.line1_y.set_data(timestamps, window_data['gravityY'])
                self.line1_z.set_data(timestamps, window_data['gravityZ'])

                # Update Acceleration subplot
                self.line2_x.set_data(timestamps, window_data['accelerationX'])
                self.line2_y.set_data(timestamps, window_data['accelerationY'])
                self.line2_z.set_data(timestamps, window_data['accelerationZ'])

                # Update Rotation Rate subplot
                self.line3_x.set_data(timestamps, window_data['rotationRateX'])
                self.line3_y.set_data(timestamps, window_data['rotationRateY'])
                self.line3_z.set_data(timestamps, window_data['rotationRateZ'])

                # Update Attitude subplot
                self.line4_roll.set_data(timestamps, window_data['attitudeRoll'])
                self.line4_pitch.set_data(timestamps, window_data['attitudePitch'])
                self.line4_yaw.set_data(timestamps, window_data['attitudeYaw'])

                # Adjust the limits and redraw the canvas
                for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                    ax.relim()
                    ax.autoscale_view()

    def start_receiver_thread(self):
        self.running = True
        self.receiver_thread = threading.Thread(target=self.receive_data)
        self.receiver_thread.start()

    def update_plot(self, frame):
        """Update the plot dynamically."""
        self.process_new_data_point()
        self.canvas.draw()

    def receive_data(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('0.0.0.0', 12345))
        server_socket.listen(1)
        print("Server is listening for incoming connections...")

        connection, client_address = server_socket.accept()
        print(f"Connection from {client_address}")

        buffer = ""
        header = True
        column_map = {}

        try:
            while self.running:
                data_chunk = connection.recv(1024).decode()
                if data_chunk:
                    buffer += data_chunk
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if header:
                            headers = line.split(',')
                            column_map = {key: index for index, key in enumerate(headers)}
                            header = False
                        else:
                            values = line.split(',')
                            if len(values) == 13:
                                with data_lock:
                                    new_row = pd.Series({
                                        'Timestamp': pd.to_datetime(float(values[column_map['Timestamp']]), unit='s'),
                                        'gravityX': float(values[column_map['gravityX']]),
                                        'gravityY': float(values[column_map['gravityY']]),
                                        'gravityZ': float(values[column_map['gravityZ']]),
                                        'accelerationX': float(values[column_map['accelerationX']]),
                                        'accelerationY': float(values[column_map['accelerationY']]),
                                        'accelerationZ': float(values[column_map['accelerationZ']]),
                                        'rotationRateX': float(values[column_map['rotationRateX']]),
                                        'rotationRateY': float(values[column_map['rotationRateY']]),
                                        'rotationRateZ': float(values[column_map['rotationRateZ']]),
                                        'attitudeRoll': float(values[column_map['attitudeRoll']]),
                                        'attitudePitch': float(values[column_map['attitudePitch']]),
                                        'attitudeYaw': float(values[column_map['attitudeYaw']])
                                    })
                                    global data
                                    data = pd.concat([data, new_row.to_frame().T], ignore_index=True)
            connection.close()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.running = False
            connection.close()
            server_socket.close()
            print("Connection closed.")

    def stop(self):
        self.running = False
        self.receiver_thread.join()

if __name__ == "__main__":
    root = tk.Tk()
    app = DynamicPlotApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()