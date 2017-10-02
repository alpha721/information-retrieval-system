import socket
import time

mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

mysocket.connect(('www.py4inf.com',80))
mysocket.send('GET http://www.py4inf.com/cover.jpg HTTP/1.0\n\n')

count = 0

picture = "";

while True:
    data = mysocket.recv(5120)
    if(len(data) < 1): break
    #time.sleep(0.25)
    count = count + len(data)

    print len(data), count
    picture = picture + data

mysocket.close()

#look for end of header 

pos = picture.find("\r\n\r\n");
print 'Header length',pos

print picture[:pos]

#skip past the header and save the picture data

picture = picture[pos + 4:]
fhand = open("stuff.jpg","wb")
fhand.write(picture);
fhand.close()
