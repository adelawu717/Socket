# 🌟Data Transaction Simulation Project

This project simulates the data transaction between an iPhone and a computer. The client and server files can now run concurrently on one computer for validation.

# 🌟How to run it
## For testing on local machine
- Step 1: add both client.py and server (serverTest.py or main.py) to run configurations
- Step 2: run the server file, and wait to see the "Server is listening for incoming connections..." on the console for socket establishment
- Step 3: run the client file. The connection is successfully built when "Connection from ('127.0.0.1', *****)" is printed on the console. Then you can move to the GUI window to see the dynamic plots.
  
## For streaming with Motion Tracking app
- Step 1: get the IP address of you laptop with command in terminal: ipconfig getifaddr en0
- Step 2: run main.py on laptop
- Step 3: on Motion Tracking app, start session on apple watch, "Start stream", and input the IP address and port number (12345)
![IMG_1349](https://github.com/user-attachments/assets/5e9f215b-cf1d-411c-861c-6b3c65f508d1)

![IMG_1350](https://github.com/user-attachments/assets/d96a7255-99c8-4006-9237-cd0641683a0b)
- Step 4: see streaming data:
<img width="1661" alt="Screenshot 2025-05-21 at 11 09 15" src="https://github.com/user-attachments/assets/ec174aad-048d-43bc-9394-4403d4aeea88" />


  
# 🌟Files Introduction

## ⌚️ Client

### `client.py`
- Sends data file line by line through a socket during a connection with the server.
- May Use the `sleep` function to simulate network delay.

## 💻 Server
### `main.py`
- run data_reciever to receive data and save data as temp.csv
- run dynamic_plot.py to read and plot temp.csv

### `data_receiver.py`
- the part where the server establishes a TCP connection with a client, and receives data from the socket and prints each line
- a temp.csv file is created to store data
  
### `dynamic_plot.py`
- Dependencies
To run this visualisation, you need to install the following Python libraries:

```sh
pip install socket
pip install threading
pip install tk
pip install matplotlib
pip install pandas
```
- read each line of the temp.csv file
- plot the data in the monitor view:
  <img width="1257" alt="Screenshot 2024-07-19 at 18 31 11" src="https://github.com/user-attachments/assets/2abb2a60-077a-4cdd-97e3-177a411c1d68">





### `newServer.py`
- **Under construction...**
- Beyond just receiving and printing out data, it will handle the data for further analysis, including:
  - Stage segmentation
  - Velocity calculation
  - Classification
  - Sample Output
  <img width="1320" alt="Screenshot 2024-07-19 at 18 36 56" src="https://github.com/user-attachments/assets/d3a75e13-8cf0-4a22-b4e3-e3f502fb10ef">
  <img width="467" alt="Screenshot 2024-07-24 at 13 03 10" src="https://github.com/user-attachments/assets/bc01fb3f-e56e-44a0-a23f-2f820a2fd60e">



## 🧮 Classifier
### `classifier.py`
- preprocess data files from Movement_data, train a RF classifier and save it
### 'classifier_model.joblib"
- the saved classifier model from classifier.py
### `Movement_data`
- The data files for training and testing

## 📃 Test file
### `GERF-L-D001-M6-S0043.csv`
- the test file of a M6 data from the Standard_data dataset
