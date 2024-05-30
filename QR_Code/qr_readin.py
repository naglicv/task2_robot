import pyqrcode
import cv2

#NOTE: This is only able to read text from qr code for now. 
#      I will try to implement to read imgs since we are gonna get the real mona lisa from QR Code 
data = "I am just too good for you guys :)"

url = pyqrcode.create(data)

url.png('myQR1.png', scale=6)

img = cv2.imread('myQR1.png')
detect = cv2.QRCodeDetector()
val, pts, st_code = detect.detectAndDecode(img)

print(val)