This project simulates the data transaction between iphone and computer. The client and server file can now run concurrently on one computer for validation. 
Client:
  client.py - send data file line by line through socket during a connection with server. use sleep function to simulate network delay
Server:
  serverTest.py - a test file which establishes a TCP connection with a client and print each line of the recieved data
  newServer.py - Under construction... Beyond just recieving and printing out data, it handle the data for further analysis: stage segmentation, velocity coculation, and classification. 
                 According to the last update, it can now print the data in-time with stage segmentation colors
