#!/usr/bin/venv python3
import modules.face_recon as face_recon
import modules.face as face
import cv2 as cv
import time

def detectAndDisplay(frame,face):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- Detect faces
    faces = face.face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = face.eyes_cascade.detectMultiScale(faceROI)
        #for (x2,y2,w2,h2) in eyes:
        #    eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #    radius = int(round((w2 + h2)*0.25))
        #    frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    cv.imshow('Capture - Face detection', frame)

if __name__ == "__main__":
    face = face_recon.Face()
    cap = cv.VideoCapture(0)#cv.VideoCapture("./data/videos/p1.mp4")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        print(frame,ret)
        detectAndDisplay(frame, face)
        if cv.waitKey(1) == 27: ## ESC
            break

    face = FacePoints(dedector_type='haar')

    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        # getting a frame
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()

        corners = face.get_points_pipeline(gray)

        if corners is not None:
            corners = np.int0(corners)
            for i in corners:
                xc,yc = i.ravel()
                cv2.circle(vis,(xc,yc),3,255,-1)

            
        # Get rectangles
        x,y,w,h = face.face_rectange
        xx,yy,ww,hh = face.eyes_rectangle

        # Draw rectangle on face
        cv2.rectangle(vis, (x,y), (x+w,y+h),(0,255,0),2)
        cv2.rectangle(vis, (xx,yy), (xx+ww,yy+hh),(0,0,255),2)

        # Show
        cv2.imshow('face track', vis)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv.destroyAllWindows()
