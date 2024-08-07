import tkinter as tk
from data_receiver import DataReceiver
from dynamic_plot import DynamicPlotApp

if __name__ == "__main__":
    # Start data receiver
    data_receiver = DataReceiver()
    data_receiver.start()

    # Keep the main thread alive to allow server to accept connections
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down server...")
        data_receiver.stop()

    # Uncomment to plot the data
    # root = tk.Tk()
    # app = DynamicPlotApp(root)
    # root.protocol("WM_DELETE_WINDOW", app.stop)
    # root.mainloop()