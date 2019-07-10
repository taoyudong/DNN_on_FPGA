# This code should be run in advance of GPU_server.py
import socket
import serial


HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 22        # Port to listen on (non-privileged ports are > 1023)

RECV_BUFFER_SIZE = 1024


def fc_on_fpga(data):
    # This function should send data to FPGA and return the received results
    ser = serial.Serial('/dev/ttyUSB1/', 19200)
    ser.write(data)
    s = ser.read(10)
    return str(s.hex())


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(RECV_BUFFER_SIZE)
            if not data:
                break
            else:
                print('Received', repr(data))
                rsts = fc_on_fpga(data)
                conn.sendall(str.encode(rsts))
