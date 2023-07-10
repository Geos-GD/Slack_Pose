# Slack_Pose

Ce projet permet d'analyser la posture d'une personne qui avance sur une slackline à l'aide de deux fichiers : AnalyseProject.py et PoseModule.py.

## AnalyseProject.py

Ce fichier contient le script principal du projet. Il utilise la bibliothèque OpenCV pour lire une vidéo d'une personne sur une slackline et effectue l'analyse de la posture en utilisant la classe poseDetector du fichier PoseModule.py. Voici les principales fonctionnalités du fichier :

- Lecture de la vidéo : Vous pouvez définir le nom de la vidéo à analyser en modifiant la variable `cap` dans le code.
- Redimensionnement de la vidéo : Vous pouvez ajuster la taille de la vidéo en modifiant la valeur de la variable `vidscale`.
- Détection de la posture : Le fichier utilise la classe poseDetector pour détecter et tracer les points clés du corps dans la vidéo. Il effectue ensuite l'analyse de la posture en utilisant les méthodes de la classe poseDetector.
- Affichage des résultats : Le fichier affiche les résultats de l'analyse de la posture, tels que la position des bras et des mains, et la posture du corps.
- Enregistrement d'images : Vous pouvez commenter/décommenter la ligne de code `cv2.imwrite` pour enregistrer ou non les images traitées.

## PoseModule.py

Ce fichier contient la classe poseDetector, qui fournit les fonctionnalités de détection de la posture. Voici les principales fonctionnalités de la classe :

- Détection de la posture : La méthode `findPose` permet de détecter et de tracer les points clés du corps dans une image.
- Détection des positions : La méthode `findPosition` récupère les coordonnées des points clés du corps détectés.
- Calcul des angles : La méthode `findAngle` permet de calculer l'angle entre trois points clés spécifiés.
- Analyse des bras : La méthode `analyseArms` analyse la position des bras et des mains et génère un diagramme en camembert pour illustrer les résultats.
- Analyse du corps : La méthode `analyseBody` analyse la posture du corps et génère un diagramme en camembert pour illustrer les résultats.
- Conversion d'images en vidéo : La méthode `jpegToMp4` permet de convertir une séquence d'images JPEG en une vidéo MP4. 

## Instructions d'utilisation

1. Assurez-vous d'avoir Python 3 installé sur votre système.
2. Installez les dépendances requises en exécutant la commande suivante : pip install opencv-python mediapipe matplotlib
3. Placez la vidéo que vous souhaitez analyser dans le répertoire `PoseVideos`.
4. Modifiez les variables appropriées dans le fichier AnalyseProject.py pour définir la vidéo à analyser et les options de redimensionnement.
5. Exécutez le fichier AnalyseProject.py en exécutant la commande suivante : python AnalyseProject.py
6. Les résultats de l'analyse de la posture seront affichés à l'écran, y compris les positions des bras et des mains, et la posture du corps.

N'hésitez pas à personnaliser le code selon vos besoins et à explorer les autres fonctionnalités disponibles dans les fichiers.

**Note :** Assurez-vous d'avoir les vidéos appropriées dans le répertoire `PoseVideos` ou de modifier les noms de fichiers dans le code pour correspondre aux vidéos que vous souhaitez analyser. La marche peut se faire en venant ou en partant mais la caméra doit être positionnée au plus proche de l’axe et dans la direction de la ligne.

Bonne analyse de posture !

![27](https://github.com/Geos-GD/Slack_Pose/assets/70376937/467a9ea4-4bac-455e-b8bd-3d23a8ab3126)

![44](https://github.com/Geos-GD/Slack_Pose/assets/70376937/6b000853-3594-496d-98fa-d82d96244412)
