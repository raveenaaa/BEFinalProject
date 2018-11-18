import os
#path = '/Users/myName/Desktop/directory'

i = 1

for file in os.listdir():
    os.rename(file, str(i)+'.jpg')
    i = i+1
