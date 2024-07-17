import socket

def receive_data():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 12345))
    server_socket.listen(1)
    print("Server is listening for incoming connections...")

    connection, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

    with connection:
        buffer = ""
        header = True
        column_map = {}
        try:
            with open('result.csv', 'w') as f:
                while True:
                    data = connection.recv(1024).decode()
                    if not data:
                        break
                    buffer += data
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
        finally:
            server_socket.close()
            print("Connection closed.")

if __name__ == "__main__":
    receive_data()