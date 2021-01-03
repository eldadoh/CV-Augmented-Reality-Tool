import cv2
import tkinter
# import tkFileDialog
from tkinter import filedialog
import os
import frame_video_convert
def faceDetector(mode='computerCamera'):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if mode=='computerCamera':
        video_capture = cv2.VideoCapture(0)
    else:
        path = filedialog.askopenfilename()
        video_capture = cv2.VideoCapture(path)
    count = 0
    while True:
        fname = str(count).zfill(4)
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70), flags=cv2.CASCADE_SCALE_IMAGE)
        i = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # if i == 0:
            #     cv2.putText(frame, "eldad", (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,255)
            # if i == 1:
            #     cv2.putText(frame, "hod", (x, y), cv2.FONT_HERSHEY_SIMPLEX,1,255)
            # i+=1
        count+=1
        cv2.imshow('Video', frame)
        cv2.imwrite(os.path.join('my_data/frames_face_detector/', fname + ".png"), frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


def computerCamera():
    faceDetector(mode='computerCamera')


def loadVideo():
    faceDetector(mode='videoFile')
if __name__ == '__main__':
    top=tkinter.Tk()
    top.title('Face detector')
    top.geometry("220x150")
    button1 = tkinter.Button(top,text= "Computer camera",command=computerCamera)
    button1.place(relx=0.5, rely=0.3, anchor='s')
    button2 = tkinter.Button(top,text= "Load video",command=loadVideo)
    button2.place(relx=0.5, rely=0.6, anchor='s')
    top.mainloop()
    frame_video_convert.image_seq_to_video('my_data/frames_face_detector', output_path='./vid2vid.mp4', fps=15.0)





