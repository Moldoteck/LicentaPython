# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 17:54:54 2018

@author: Cristian
"""

import struct
import socket               # Import socket module

s = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
port = 12346                # Reserve a port for your service.

s.connect((host, port))

with open('D:\\Desktop all\\forlearn\\img\\46.jpg', 'rb') as img:
    img_str = img.read()
    
    print('len:', len(img_str))
    
    # send string size
    tip = struct.pack('!i', 0)
    len_str = struct.pack('!i', len(img_str))
    s.send(tip)
    s.send(len_str)
  
    # send string image
    
    s.send(img_str)
    #print(s.recv(1024))
    s.close 