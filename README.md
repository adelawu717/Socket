# Data Transaction Simulation Project

This project simulates the data transaction between an iPhone and a computer. The client and server files can now run concurrently on one computer for validation.

## Client

### `client.py`
- Sends data file line by line through a socket during a connection with the server.
- Uses the `sleep` function to simulate network delay.

## Server

### `serverTest.py`
- A test file which establishes a TCP connection with a client.
- Prints each line of the received data.

### `newServer.py`
- **Under construction...**
- Beyond just receiving and printing out data, it will handle the data for further analysis, including:
  - Stage segmentation
  - Velocity calculation
  - Classification
- According to the last update, it can now print the data in real-time with stage segmentation colors.

