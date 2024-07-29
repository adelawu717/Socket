import socket
import threading
import pandas as pd
import os

# Global variable and lock for data
data_lock = threading.Lock()
file_path = 'temp.csv'

class DataReceiver:
    def __init__(self, host='0.0.0.0', port=12345, timeout=5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.running = False

        # Initialize the CSV file if it doesn't exist
        self.initialize_csv()

    def initialize_csv(self):
        if not os.path.exists(file_path):
            # Define the columns for the CSV file
            columns = [
                'Timestamp', 'gravityX', 'gravityY', 'gravityZ',
                'accelerationX', 'accelerationY', 'accelerationZ',
                'rotationRateX', 'rotationRateY', 'rotationRateZ',
                'attitudeRoll', 'attitudePitch', 'attitudeYaw'
            ]
            # Create an empty DataFrame with the specified columns
            data = pd.DataFrame(columns=columns)
            # Save the empty DataFrame to a CSV file
            data.to_csv(file_path, index=False)
            print(f"Initialized {file_path} with headers.")

    def start(self):
        self.running = True
        self.receiver_thread = threading.Thread(target=self.receive_data)
        self.receiver_thread.start()

    def receive_data(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)
        print("Server is listening for incoming connections...")

        connection, client_address = server_socket.accept()
        connection.settimeout(self.timeout)
        print(f"Connection from {client_address}")

        buffer = ""
        header = True
        column_map = {}

        try:
            while self.running:
                try:
                    data_chunk = connection.recv(1024).decode()
                    if data_chunk:
                        buffer += data_chunk
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            print(f"Received line: {line}")  # Print each received line
                            if header:
                                headers = line.split(',')
                                column_map = {key: index for index, key in enumerate(headers)}
                                header = False
                                with open(file_path, 'w') as f:
                                    f.write(','.join(headers) + '\n')
                            else:
                                values = line.split(',')
                                if len(values) == 13:
                                    with data_lock:
                                        with open(file_path, 'a') as f:
                                            f.write(line + '\n')
                    else:
                        break
                except socket.timeout:
                    print("Socket timed out, no data received.")
                    break
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
