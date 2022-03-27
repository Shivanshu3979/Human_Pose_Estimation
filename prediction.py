import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils # For drawing keypoints
points = mpPose.PoseLandmark # Landmarks
test = "DATASET/TEST/"


def get_prediction(results,model):
    return model.predict(results)

data=pd.read_csv('data.csv')
num_of_classes=len(data.pose.unique())
x=data.drop(axis=0, columns=['pose'])
y=data.pose

print(x.shape)
print(y.shape)
data_train,data_test,y_train,y_test=train_test_split(x,y,
                                                     shuffle=True,
                                                    test_size=0.4,
                                                    random_state=1)
evalset = [(data_train, y_train), (data_test,y_test)]
print("Data Train:",data_train.shape)
print("Data Test:",data_test.shape)
print("label Train:",y_train.shape)
print("label Test:",y_test.shape)

xgb = XGBClassifier(booster='gbtree', objective='multi:softprob', random_state=42, eval_metric="mlogloss", num_class=num_of_classes)
xgb.fit(data_train,y_train, eval_set=evalset)


font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 100)
fontScale = 4
color = (255, 0, 0)
thickness = 4

data = []
data.append("")
data.append("pose")
for p in points:
        x = str(p)[13:]
        data.append(x + "_x")
        data.append(x + "_y")
        data.append(x + "_z")
        data.append(x + "_vis")
head=data;
data = pd.DataFrame(columns = data) # Empty dataset

ch=int(input("[1] read from Test set\n[2] read from Camera frame\nChoice: "))

if ch==1:
    for folder in os.listdir(test):
        img=cv2.imread(test+folder+"/"+os.listdir(test+folder)[np.random.randint(len(os.listdir(test+folder)))])
        temp = []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imageWidth, imageHeight = img.shape[:2]
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blackie = np.zeros(img.shape) # Blank image
        results = pose.process(imgRGB)
        if results.pose_landmarks:
                # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) #draw landmarks on image
                mpDraw.draw_landmarks(blackie, results.pose_landmarks, mpPose.POSE_CONNECTIONS) # draw landmarks on blackie
                landmarks = results.pose_landmarks.landmark
                temp=temp+[0,folder]
                for i,j in zip(points,landmarks):
                        temp = temp + [j.x, j.y, j.z, j.visibility]
                data.loc[0] = temp
                data.loc[1] = data.loc[0]
                x=data.drop(axis=0, columns=['pose'])
                y=data.pose
                _,_t,y_tr,y_te=train_test_split(x,y,
                                                shuffle=True,
                                                test_size=0.5,
                                                random_state=1)
                string=get_prediction(_t,xgb)
                blackie=cv2.putText(blackie, string[0], org, font,fontScale, color, thickness, cv2.LINE_AA)
                fig=plt.figure(figsize=(18, 5), dpi=80)
                fig.add_subplot(1,2,1)
                plt.imshow(img)

                fig.add_subplot(1,2,2)
                plt.imshow(blackie)
                plt.show()


elif ch==2:
    cap=cv2.VideoCapture(0)
    while True:
        temp = []
        ret, img = cap.read()
        img=cv2.flip(img, 1,0)
        results = pose.process(img)
        if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) #draw landmarks on image
                #mpDraw.draw_landmarks(blackie, results.pose_landmarks, mpPose.POSE_CONNECTIONS) # draw landmarks on blackie
                landmarks = results.pose_landmarks.landmark
                temp=temp+[0,"NULL"]
                for i,j in zip(points,landmarks):
                        temp = temp + [j.x, j.y, j.z, j.visibility]
                data.loc[0] = temp
                data.loc[1] = data.loc[0]
                x=data.drop(axis=0, columns=['pose'])
                y=data.pose
                _,_t,y_tr,y_te=train_test_split(x,y,
                                                shuffle=True,
                                                test_size=0.5,
                                                random_state=1)
                string=get_prediction(_t,xgb)
                img=cv2.putText(img, string[0], org, font,fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow("abc",img)
        if cv2.waitKey(10) == ord('q'):  # wait a bit, and see keyboard press
          break                        # if q pressed, quit

    # release things before quiting
    cap.release()
    cv2.destroyAllWindows()
