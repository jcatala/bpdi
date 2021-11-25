#!/usr/bin/venv python3
import modules.face_recon as face_recon
import cv2 as cv

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
    cap = cv.VideoCapture("./data/videos/p1.mp4")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        print(frame,ret)
        detectAndDisplay(frame, face)
        if cv.waitKey(10) == 27:
            break

    cap.release()
    cv.destroyAllWindows()
    print("deadbeef")
