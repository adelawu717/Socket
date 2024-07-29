import tkinter as tk
from data_receiver import DataReceiver
from dynamic_plot import DynamicPlotApp

if __name__ == "__main__":
    # Start data receiver
    data_receiver = DataReceiver()
    data_receiver.start()

    # Wait for the data receiver to finish
    data_receiver.receiver_thread.join()

    # Uncomment to plot the data
    # root = tk.Tk()
    # app = DynamicPlotApp(root)
    # root.protocol("WM_DELETE_WINDOW", app.stop)
    # root.mainloop()
