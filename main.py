import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import time
import scipy

from peakutils import indexes
from scipy.signal import find_peaks

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0) 

stop_button = False

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

time_values = [0]
green_values = []
start_time = time.time()

def stop_capture(event):
    global stop_button
    stop_button = True

fig.canvas.mpl_connect('close_event', stop_capture)

def animate(i):
    ax1.clear()
    ax1.plot(time_values, green_values, 'g-', color='green')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Green Value')
    ax1.set_title('Green Value over Time')



while not stop_button:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale (gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:
            cv2. rectangle (roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0))
           
        forehead_y = int(y + h * 0.05)
        forehead_h = int(h * 0.15)
        forehead_width = int(w * 0.5)
        forehead_x = int(x + (w - forehead_width) / 2)

        forehead_roi = img[forehead_y:forehead_y+forehead_h, forehead_x:forehead_x+forehead_width]
        forehead_green = np.mean(forehead_roi[:, :, 1])

        cv2.rectangle(img, (forehead_x, forehead_y), (forehead_x+forehead_width, forehead_y + forehead_h), (80,127,255), 2)
        cv2.putText(img, f"Green Value: {forehead_green:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        #Green Data Plot
        current_time = time.time() - start_time
        time_values.append(current_time)
        green_values.append(forehead_green)

        # ani = animation.FuncAnimation(fig, animate, interval=1000)
        plt.tight_layout()  
        
        # chart_img = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        # chart_img = chart_img.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        # chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGB2BGR)

    
        # chart_img = cv2.resize(chart_img, (img.shape[1], int(img.shape[0] * 0.3)))  # Adjust the size as needed

        
        # combined_img = np.vstack((img, chart_img))

    # cv2.imshow('combined_img', combined_img)

    #Peak Finding Algorithm
    value = np.array(green_values)
    # heartRate = []

    peaks, _ = find_peaks(value, height=5, prominence=None, distance=None, width=None, threshold=None, rel_height=0.5, wlen=None)
    plt.plot(value, color = 'green')
    plt.plot(peaks, value[peaks], "o", markersize = 10, color='red')

    value_time = np.array(time_values)
    num_peaks = len(peaks)
    basic_time = max(value_time)


  
    print("Number of ime:", basic_time)
    
    if basic_time > 0:
        heart_rate = (np.mean(num_peaks/basic_time)) * 120
    else:
        heart_rate = 1


    cv2.putText(img, f"Heart Rate: {heart_rate:.2f}", (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('img', img)
    plt.pause(0.1)

    k = cv2.waitKey(30) & 0xFF

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()