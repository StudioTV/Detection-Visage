# MIT License
# 
# Copyright (c) 2019 StudioTV at youtube.com/studiotv
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2

cascadeClassifierPath = 'haarcascade_frontalface_alt.xml' # Chemin du Classifier
cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)
cap = cv2.VideoCapture("video.mp4") # On récupère la vidéo

while(cap.isOpened()):
	_, frame = cap.read()
	grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Conversion N/B
	detectedFaces = cascadeClassifier.detectMultiScale(grayImage,  scaleFactor=1.1, minNeighbors=10, minSize=(20, 20)) # Détection

	for(x,y, width, height) in detectedFaces:
		cv2.rectangle(frame, (x, y), (x+width, y+height), (0,255,0), 3) # Dessin d'un rectangle
		
	cv2.imshow("result", frame)
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

# RÉGLER VOS PARAMÈTRES
#
# scaleFactor: Certains visages n'auront pas la même taille sur l'image, ce paramètre permet de compenser ce phénomène.
# 
# 				Intervalle : 1.1 - 1.4
# 					Basse -> L'algorithme sera lent, mais aura plus de détection.
# 					Haute -> L'algorithme sera plus rapide, au risque de perdre des détections.
# 
# minNeighbors: Cette valeur permet de spécifier combien de voisin doit avoir chaque rectangle de détection pour prendre en compte
#					ce potentiel visage.
# 				Une valeur haute donnera donc moins de détection, mais les détections seront d'une meilleure qualité !	
#				En augmentant ce nombre, vous éliminez les faux positifs, mais soyez prudents, vous pouvez également 
# 					perdre des vrais positifs.
#  
# minSize: Les objets inférieurs à cette taille ne seront pas pris en compte 
#  
# maxSize: Les objets supérieurs à cette taille ne seront pas pris en compte 