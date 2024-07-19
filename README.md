# Data Transaction Simulation Project

This project simulates the data transaction between an iPhone and a computer. The client and server files can now run concurrently on one computer for validation.

# How to run it
- Step 1: add both client and server (serverTest or newServer) to run configurations
- Step 2: run the server file, and wait to see the "Server is listening for incoming connections..." on the console for socket establishment
- Step 3: run the client file. The connection is successfully built when "Connection from ('127.0.0.1', 53701)" is printed on the console. Then you can move to the GUI window to see the dynamic plots.
  
## Files Introduction

## Client

### `client.py`
- Sends data file line by line through a socket during a connection with the server.
- Uses the `sleep` function to simulate network delay.

## Server

### `serverTest.py`
- A test file which establishes a TCP connection with a client.
- Prints each line of the received data.
- Sample Output
  <img width="1257" alt="Screenshot 2024-07-19 at 18 31 11" src="https://github.com/user-attachments/assets/2abb2a60-077a-4cdd-97e3-177a411c1d68">


### `newServer.py`
- **Under construction...**
- Beyond just receiving and printing out data, it will handle the data for further analysis, including:
  - Stage segmentation
  - Velocity calculation
  - Classification
  - Sample Output
  <img width="1320" alt="Screenshot 2024-07-19 at 18 36 56" src="https://github.com/user-attachments/assets/d3a75e13-8cf0-4a22-b4e3-e3f502fb10ef">


## Classifier
### `classifier.py`
- preprocess data files from Movement_data, train a RF classifier and save it
###'classifier_model.joblib"
- the saved classifier model from classifier.py
### `Movement_data`
- The data files for training and testing

## Test file
### `GERF-L-D001-M6-S0043.csv`
- the test file of a M6 data from the Standard_data dataset
