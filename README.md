# Human Detection & Motion Alert System using OpenCV and Telegram API

This project is a **real-time human detection and motion alert system** built using Python, OpenCV, and the Telegram API.  
It detects humans from webcam footage, highlights them with bounding boxes, plays an alert sound, sends an instant Telegram notification, and saves snapshots of detected motion.

---

## Features

- ✅ **Real-Time Human Detection** using MobileNet SSD with OpenCV's DNN module  
- ✅ **Motion Snapshot Saving** in a dedicated folder  
- ✅ **Beep Sound Alert** on human detection  
- ✅ **Telegram Notification Integration** for remote alerts  
- ✅ **Multi-threading** for seamless operation without UI lag  

---

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- imutils
- numpy
- requests
- winsound (Windows only)

### Install Dependencies:
```bash
pip install opencv-python imutils numpy requests
