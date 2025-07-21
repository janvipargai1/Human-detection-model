import cv2
import time
import imutils
import numpy as np
import threading
import winsound
import os
import requests


prototxt=r"C:\Users\HP\Desktop\opencv project\deploy.prototxt"
model=r"C:\Users\HP\Desktop\opencv project\mobilenet_iter_73000.caffemodel"

neutral_network=cv2.dnn.readNetFromCaffe(prototxt,model)

#initialise webcam
cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

#motion detection parameter
area=1000
motion_detected=False
recording=False
video_writer=None
video_folder="Motion_Videos"
os.makedirs(video_folder,exist_ok=True)

#telegram alert api
telegram_bot_token="8071060029:AAEZFh7R4Ghl4QScNOeJvZREiud22bkHZyY"
telegram_chat_id="5159017252"

#function to send an alert to telegram
def send_telegram_alert(message):
    url=f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    data={"chat_id":telegram_chat_id,"text":message}
    requests.post(url,data=data)
    
    
#function to play a beep sound
def play_beep():
    winsound.Beep(5000,2000)
    
    
while True:
    ret,img=cam.read()
    if not ret:
        break
    
    img=imutils.resize(img,width=500)
    (h,w)=img.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),0.007843,(300,300),127.7)
    
    #object detection
    neutral_network.setInput(blob)
    detection=neutral_network.forward()
    detected_human=False
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.6:  # Confidence threshold
            class_id = int(detection[0, 0, i, 1])
            if class_id == 15:  # Class 15 = "Person"
                detected_human = True
                box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x_max, y_max) = box.astype("int")
                cv2.rectangle(img, (x, y), (x_max, y_max), (0, 255, 0), 2)
                
                
    # Motion Detection Logic
    text = "No Motion"
    if detected_human:
        text = "Human Detected"
        motion_detected = True
        
    # Save a snapshot
    snapshot_path = os.path.join(video_folder, f"motion_{int(time.time())}.jpg")
    cv2.imwrite(snapshot_path, img)
    
    # Send alerts (multi-threading to avoid lag)
    threading.Thread(target=play_beep).start()
    threading.Thread(target=send_telegram_alert, args=("🚨 Motion detected! Check your feed.",)).start()
    
    # Display text on screen
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Camera Feed", img)
    
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    
    # Release resources
cam.release()
cv2.destroyAllWindows()