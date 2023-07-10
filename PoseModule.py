# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 18:50:50 2023

@author: Daenzer Guillaume
"""

import cv2
import mediapipe as mp
import time
import math
import matplotlib.pyplot as plt
import textwrap
import os
import os.path
from math import sqrt


class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
        #                              self.detectionCon, self.trackCon)
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth
                                 , min_detection_confidence=0.5
                                 , min_tracking_confidence=0.5
                                 )
        self.armPie = []
        self.bodyPie = []
        self.tolerance = 0

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
    

    # self.pie prends 1 ligne par image : brasG, brasD, mainG, mainD. Si bras sont haut : 1, sinon 0. Si mains sont exterieurs : 1, sinon 0. Ex : [1,1,1,1]
    def analyseArms(self, img, direction, brasG =True, brasD =True, mainG = True, mainD = True, draw=True):
        txtYcount = 50
        linepie = []
        #########################################################################################################################################################
        ###############################################          DIRECTION = COME               ################################################################
        #########################################################################################################################################################
        if direction == 'come' :
            if brasG :
                if len(self.lmList) != 0 :
                    if self.lmList[11][2] < self.lmList[19][2] :
                        # print("Bras gauche vers le bas")
                        linepie.append(0)
                        if draw:
                            cv2.putText(img, "Bras gauche vers le bas", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 247), 2)
                            cv2.line(img, (self.lmList[11][1], self.lmList[11][2]), (self.lmList[19][1], self.lmList[19][2]), 
                                     (0, 255, 247), 10)
                    else : 
                        linepie.append(1)
                        if draw:
                            cv2.putText(img, "Bras gauche vers le haut", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (50, 0, 255), 2)
                            cv2.line(img, (self.lmList[11][1], self.lmList[11][2]), (self.lmList[19][1], self.lmList[19][2]), 
                                     (50, 0, 255), 10)
                    txtYcount +=50
                
            if brasD :
                if len(self.lmList) != 0 :
                    if self.lmList[12][2] < self.lmList[20][2] :
                        linepie.append(0)
                        # print("Bras droite vers le bas")
                        if draw:
                            cv2.putText(img, "Bras droit vers le bas", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 247), 2)
                            cv2.line(img, (self.lmList[12][1], self.lmList[12][2]), (self.lmList[20][1], self.lmList[20][2]), 
                                     (0, 255, 247), 10)
                    else : 
                        linepie.append(1)
                        if draw:
                            cv2.putText(img, "Bras droit vers le haut", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (50, 0, 250), 2)
                            cv2.line(img, (self.lmList[12][1], self.lmList[12][2]), (self.lmList[20][1], self.lmList[20][2]), 
                                     (50, 0, 255), 10)
                    txtYcount +=50
                    
            if mainG :
                if len(self.lmList) != 0 :
                    if self.lmList[13][1] > self.lmList[19][1] :
                        linepie.append(0)
                        if draw:
                            cv2.putText(img, "Main gauche vers linterieur", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
                            
                            cv2.circle(img, (self.lmList[19][1], self.lmList[19][2]), 10, (255, 255, 0), 2)
                            cv2.circle(img, (self.lmList[19][1], self.lmList[19][2]), 15, (255, 255, 0), 2)
                            
        
                            cv2.line(img, (self.lmList[13][1], self.lmList[13][2]-100), (self.lmList[13][1], self.lmList[13][2]+100), 
                                     (255, 255, 0), 4)
                    else : 
                        linepie.append(1)
                        if draw:
                            cv2.putText(img, "Main gauche vers l'exterieur", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                            cv2.circle(img, (self.lmList[19][1], self.lmList[19][2]), 10, (255, 0, 255), 2)
                            cv2.circle(img, (self.lmList[19][1], self.lmList[19][2]), 15, (255, 0, 255), 2)
                    
                    txtYcount +=50
                    
                    
            if mainD :
                if len(self.lmList) != 0 :
                    if self.lmList[14][1] < self.lmList[20][1] :
                        linepie.append(0)
                        if draw:
                            cv2.putText(img, "Main droite vers linterieur", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
                            
                            cv2.circle(img, (self.lmList[20][1], self.lmList[20][2]), 10, (255, 255, 0), 2)
                            cv2.circle(img, (self.lmList[20][1], self.lmList[20][2]), 15, (255, 255, 0), 2)
                            
        
                            cv2.line(img, (self.lmList[14][1], self.lmList[14][2]-100), (self.lmList[14][1], self.lmList[14][2]+100), 
                                     (255, 255, 0), 4)
                    else : 
                        linepie.append(1)
                        if draw:
                            cv2.putText(img, "Main droite vers l'exterieur", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                            cv2.circle(img, (self.lmList[20][1], self.lmList[20][2]), 10, (255, 0, 255), 2)
                            cv2.circle(img, (self.lmList[20][1], self.lmList[20][2]), 15, (255, 0, 255), 2)
                    
                    txtYcount +=50
                    
        #################################################################################################################################################################
        #######################################################          DIRECTION = GO               #################################################################
        #################################################################################################################################################################
        if direction == 'go' :
            if brasG :
                if len(self.lmList) != 0 :
                    if self.lmList[11][2] < self.lmList[19][2] :
                        linepie.append(0)
                        # print("Bras gauche vers le bas")
                        if draw:
                            cv2.putText(img, "Bras gauche vers le bas", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 247), 2)
                            cv2.line(img, (self.lmList[11][1], self.lmList[11][2]), (self.lmList[19][1], self.lmList[19][2]), 
                                     (0, 255, 247), 10)
                    else : 
                        linepie.append(1)
                        if draw:
                            cv2.putText(img, "Bras gauche vers le haut", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (50, 0, 255), 2)
                            cv2.line(img, (self.lmList[11][1], self.lmList[11][2]), (self.lmList[19][1], self.lmList[19][2]), 
                                     (50, 0, 255), 10)
                    txtYcount +=50
                
            if brasD :
                if len(self.lmList) != 0 :
                    if self.lmList[12][2] < self.lmList[20][2] :
                        linepie.append(0)
                        # print("Bras droite vers le bas")
                        if draw:
                            cv2.putText(img, "Bras droit vers le bas", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 247), 2)
                            cv2.line(img, (self.lmList[12][1], self.lmList[12][2]), (self.lmList[20][1], self.lmList[20][2]), 
                                     (0, 255, 247), 10)
                    else : 
                        linepie.append(1)
                        if draw:
                            cv2.putText(img, "Bras droit vers le haut", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (50, 0, 250), 2)
                            cv2.line(img, (self.lmList[12][1], self.lmList[12][2]), (self.lmList[20][1], self.lmList[20][2]), 
                                     (50, 0, 255), 10)
                    txtYcount +=50
                    
            if mainG :
                if len(self.lmList) != 0 :
                    if self.lmList[13][1] < self.lmList[19][1] :
                        linepie.append(0)
                        if draw:
                            cv2.putText(img, "Main gauche vers linterieur", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
                            
                            cv2.circle(img, (self.lmList[19][1], self.lmList[19][2]), 10, (255, 255, 0), 2)
                            cv2.circle(img, (self.lmList[19][1], self.lmList[19][2]), 15, (255, 255, 0), 2)
                            
        
                            cv2.line(img, (self.lmList[13][1], self.lmList[13][2]-100), (self.lmList[13][1], self.lmList[13][2]+100), 
                                     (255, 255, 0), 4)
                    else : 
                        linepie.append(1)
                        if draw:
                            cv2.putText(img, "Main gauche vers l'exterieur", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                            cv2.circle(img, (self.lmList[19][1], self.lmList[19][2]), 10, (255, 0, 255), 2)
                            cv2.circle(img, (self.lmList[19][1], self.lmList[19][2]), 15, (255, 0, 255), 2)
                    
                    txtYcount +=50
                    
                    
            if mainD :
                if len(self.lmList) != 0 :
                    if self.lmList[14][1] > self.lmList[20][1] :
                        linepie.append(0)
                        if draw:
                            cv2.putText(img, "Main droite vers linterieur", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
                            
                            cv2.circle(img, (self.lmList[20][1], self.lmList[20][2]), 10, (255, 255, 0), 2)
                            cv2.circle(img, (self.lmList[20][1], self.lmList[20][2]), 15, (255, 255, 0), 2)
                            
        
                            cv2.line(img, (self.lmList[14][1], self.lmList[14][2]-100), (self.lmList[14][1], self.lmList[14][2]+100), 
                                     (255, 255, 0), 4)
                    else : 
                        linepie.append(1)
                        if draw:
                            cv2.putText(img, "Main droite vers l'exterieur", (50, txtYcount),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                            cv2.circle(img, (self.lmList[20][1], self.lmList[20][2]), 10, (255, 0, 255), 2)
                            cv2.circle(img, (self.lmList[20][1], self.lmList[20][2]), 15, (255, 0, 255), 2)
                    
                    txtYcount +=50
        self.armPie.append(linepie)
    

 
    def sqr(self,a):
        return a*a
     
    def distance(self,x1,y1,x2,y2):
        return sqrt(self.sqr(y2-y1)+self.sqr(x2-x1))    
    
    def analyseBody(self, img, tolerance, draw=True):
        self.tolerance = tolerance
        # txtX = img.shape[1]-int(img.shape[1]/2)
        linepie = []
        
        if len(self.lmList) != 0 :
            x12, y12 = self.lmList[12][1:]
            x24, y24 = self.lmList[24][1:]
            x11, y11 = self.lmList[11][1:]
            x23, y23 = self.lmList[23][1:]
            dist_12_24 = self.distance(x12, y12, x24, y24)
            dist_11_23 = self.distance(x11, y11, x23, y23)
        
        if len(self.lmList) != 0 :
            if dist_11_23 > dist_12_24+(dist_12_24/100*tolerance)  :
                linepie.append(0)
                if draw:
                    cv2.putText(img, "Corps penche a droite", (50, 300),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.line(img, (self.lmList[12][1], self.lmList[12][2]), (self.lmList[24][1], self.lmList[24][2]), 
                             (255, 255, 255), 10)
            elif dist_12_24 > dist_11_23+(dist_11_23/100*tolerance) : 
                linepie.append(1)
                if draw:
                    cv2.putText(img, "Corps penche a gauche", (50, 300),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.line(img, (self.lmList[11][1], self.lmList[11][2]), (self.lmList[23][1], self.lmList[23][2]), 
                                (255, 255, 255), 10)
            else : 
                linepie.append(2)
                if draw:
                    cv2.putText(img, "Corps au centre", (50, 300),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.line(img, (self.lmList[11][1], self.lmList[11][2]), (self.lmList[23][1], self.lmList[23][2]), 
                             (255, 255, 255), 10)
                    cv2.line(img, (self.lmList[12][1], self.lmList[12][2]), (self.lmList[24][1], self.lmList[24][2]), 
                             (255, 255, 255), 10)
                        
        self.bodyPie.append(linepie)
        
        
        
        
    def printArmPie(self):
        print (self.armPie)
        
    def translateArmLabel(self, category):
        labels = ["brasG en bas", "brasG en haut", "brasD en bas", "brasD en haut", "mainG intérieur",
                  "mainG extérieur", "mainD intérieur", "mainD extérieur"]
        translated_labels = []
        for i in range(4):
            if i < len(category) and category[i] == 0:
                translated_labels.append(labels[2*i])
            else:
                translated_labels.append(labels[2*i+1])
        return ", ".join(translated_labels)
    
    def plotArmPie(self):
        # Comptage des occurrences de chaque catégorie dans la liste
        counts = {}
        for sublist in self.armPie:
            key = tuple(sublist)
            counts[key] = counts.get(key, 0) + 1
    
        # Extraction des catégories et de leurs occurrences
        categories = []
        occurrences = []
        for key, count in counts.items():
            categories.append(key)
            occurrences.append(count)
    
        # Conversion des catégories en libellés traduits
        translated_categories = [self.translateArmLabel(category) for category in categories]
        wrapped_labels = []
        for translated_category in translated_categories:
            wrapped_label = textwrap.fill(translated_category, 17)  # Met à la ligne si plus de 17 caractères
            wrapped_labels.append(wrapped_label)

        # Tracé du graphique en camembert (pie chart) avec les libellés mis à la ligne
        plt.figure() # Create a new figure
        plt.pie(occurrences, labels=wrapped_labels, autopct='%1.1f%%')
        plt.title('Répartition de la position des bras et des mains')
        plt.axis('equal')  # Aspect ratio égal pour un cercle
        plt.show()
        
    
    def plotBodyPie(self):
        # print (self.bodyPie)
        labels = ['Corps à droite', 'Corps à gauche', 'Corps au centre']
        counts = [0, 0, 0]
        for linepie in self.bodyPie :
            for value in linepie:
                counts[value] += 1
        plt.figure() # Create a new figure
        plt.pie(counts, labels=labels, autopct='%1.1f%%')
        plt.title('Répartition de la posture du corps, tolérance = '+ str(self.tolerance) +' %')
        plt.show()
        
        
def jpegToMp4 (output):
    # Arguments
    dir_path = 'Images'
    images = []
    for f in sorted(os.listdir(dir_path), key=lambda x: int(x.split('.')[0])):
        if f.endswith('.jpeg'):
            images.append(f)
            # print(f)

    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    #cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    for image in images:

        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)

        out.write(frame) # Write out frame to video

    # Release everything if job is finished
    out.release()

        
       
def main():
    cap = cv2.VideoCapture('PoseVideos/1.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()