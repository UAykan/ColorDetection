
import cv2
from collections import deque
import IP
import time

from sceneRadianceHE import RecoverHE
from sceneRadianceCLAHE import RecoverCLAHE


def nothing(x):
    pass

buffer_size = 16
pts = deque(maxlen = buffer_size)

cap = cv2.VideoCapture("C:/auv/ColorDetection/video.avi")
cap.set(3,960)
cap.set(4,480)

cap.set(4,480)
prev_frame_time = 0
new_frame_time = 0
while cap.isOpened():  
    success,imgOriginal = cap.read()
    
                   
    if success:  
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(fps)
        print("FPS: "+fps)       
        
        #scene=RecoverHE(imgOriginal)
        scene=RecoverCLAHE(imgOriginal)
        #rscene=cv2.resize(scene,(960,480))
        
        contours,mask=IP.process_img(scene) 
        #contours,mask=IP.process_img(imgOriginal)
        center = None
    
        if len(contours) > 0:            
           center=IP.draw_img(contours, imgOriginal)
            
        pts.appendleft(center)
        imgOriginal=cv2.resize(imgOriginal,(960,480))
        mask=cv2.resize(mask,(960,480))
        
        cv2.imshow("Orijinal Tespit",imgOriginal)
        cv2.imshow("Maskeli Tespit",mask)
    
        if cv2.waitKey(1)==ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()


