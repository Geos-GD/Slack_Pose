# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 18:51:15 2023

@author: Daenzer Guillaume
"""

import cv2
import time
import PoseModule as pm


# idées : ajouter des warnings, ajout une fct de prévision de chute si centre de gravité des points du corps est éloigné du centre des pieds
#####################################################################################################################################################################################################
################################################################### Set video name, direction from camera and video scale ###########################################################################
# # EXPERIENCE 1 - Guillaume Serre Ponçon
# cap = cv2.VideoCapture('PoseVideos/6.mp4')
# direction = 'come'
# vidscale = 1
# EXPERIENCE 2 - Raphaël L'Hongrin
cap = cv2.VideoCapture('PoseVideos/7-2.mp4')
direction = 'go'
vidscale = 0.8
#####################################################################################################################################################################################################
pTime = 0
detector = pm.poseDetector()
cnt = 1
while True:
    success, img = cap.read()
    if not success:
        break
    if vidscale != 1 :
        img = cv2.resize(img,(0,0),fx=vidscale,fy=vidscale)
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    ################################################################### ANALYSE ####################################################################################################################
    detector.analyseArms(img, direction = direction, brasG=True, brasD=True, mainG=True, mainD=True, draw=True)
    detector.analyseBody(img, tolerance=15, draw=True) # tolerance en % de différence de longueur entre côté droit et gauche pour que le corps soit considéré comme droit 
    ################################################################################################################################################################################################
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, 'fps :'+ str(int(fps)), (70, img.shape[0]-50), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0), 3) # Plot fps
    cv2.imshow("Image", img)
    ################################################################### SAVE IMAGES #################################################################################################################
    # cv2.imwrite('Images/'+str(cnt)+'.jpeg',img)
    #################################################################################################################################################################################################
    cnt+=1
    # Sortir de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
################################################################### PLOT PIES #######################################################################################################################
detector.plotArmPie()
detector.plotBodyPie()

################################################################### RECOMPOSE VIDEO FROM JPEG IMAGES ################################################################################################
# pm.jpegToMp4('Raph.mp4')
#####################################################################################################################################################################################################

# Arrêter la capture vidéo et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()