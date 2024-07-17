import socket
import time

def send_data(host, port, filepath):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        with open(filepath, 'r') as file:
            # Send header to help the server identify columns (optional)
            headers = next(file)
            sock.sendall(headers.encode())
            # Send data
            for line in file:
                sock.sendall(line.encode())
                time.sleep(0.1)  # Simulate network delay

if __name__ == "__main__":
    HOST, PORT = 'localhost', 12345
    FILEPATH = 'GERF-L-D001-M6-S0043.csv'
    send_data(HOST, PORT, FILEPATH)