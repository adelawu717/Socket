import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import pandas as pd

file_path = 'temp.csv'


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

        # Set up the animation with a slower update interval (e.g., 1000 ms)
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=1000)

    def update_plot(self, frame):
        """Update the plot with data from the CSV file."""
        try:
            data = pd.read_csv(file_path)
            data['Timestamp'] = pd.to_numeric(data['Timestamp'], errors='coerce')
            data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
        except pd.errors.EmptyDataError:
            return
        except ValueError as e:
            print(f"Error parsing datetime: {e}")
            return

        if not data.empty:
            print("Updating plot with new data...")  # Debug statement

            timestamps = mdates.date2num(data['Timestamp'])

            # Update Gravity subplot
            self.line1_x.set_data(timestamps, data['gravityX'])
            self.line1_y.set_data(timestamps, data['gravityY'])
            self.line1_z.set_data(timestamps, data['gravityZ'])

            # Update Acceleration subplot
            self.line2_x.set_data(timestamps, data['accelerationX'])
            self.line2_y.set_data(timestamps, data['accelerationY'])
            self.line2_z.set_data(timestamps, data['accelerationZ'])

            # Update Rotation Rate subplot
            self.line3_x.set_data(timestamps, data['rotationRateX'])
            self.line3_y.set_data(timestamps, data['rotationRateY'])
            self.line3_z.set_data(timestamps, data['rotationRateZ'])

            # Update Attitude subplot
            self.line4_roll.set_data(timestamps, data['attitudeRoll'])
            self.line4_pitch.set_data(timestamps, data['attitudePitch'])
            self.line4_yaw.set_data(timestamps, data['attitudeYaw'])

            # Adjust the limits and redraw the canvas
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.relim()
                ax.autoscale_view()

            self.canvas.draw()

    def stop(self):
        self.root.quit()
