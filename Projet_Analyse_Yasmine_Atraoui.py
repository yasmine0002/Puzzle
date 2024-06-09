##Analyse d'image
##projet Yasmine ATRAOUI

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carica l'immagine
image = cv2.imread('puzzle.jpg', cv2.IMREAD_GRAYSCALE)

# Applica un filtraggio di mediana per ridurre il rumore
image_filtered = cv2.medianBlur(image, 5)

# Applica una correzione del contrasto
alpha = 1.5  # Fattore di contrasto
beta = 0     # Correzione di luminosità (0 per non cambiare)
image_contrast = cv2.convertScaleAbs(image_filtered, alpha=alpha, beta=beta)

# Applica una binarizzazione adattiva
thresh = cv2.adaptiveThreshold(image_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

# Applica operazioni morfologiche per pulire l'immagine binarizzata
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


kernel = np.ones((2,2),np.uint8)
thresh_dilated = cv2.dilate(thresh, kernel, iterations=1)

kernel = np.ones((2,2), np.uint8)  # Definisci il kernel di erosione
thresh_eroded = cv2.erode(thresh_dilated, kernel, iterations=20)  # Applica l'erosione


kernel = np.ones((2,2),np.uint8)
thresh_dilated = cv2.dilate(thresh, kernel, iterations=3)

kernel = np.ones((2,2), np.uint8)  # Definisci il kernel di erosione
thresh_eroded = cv2.erode(thresh_dilated, kernel, iterations=3)  # Applica l'erosione

kernel = np.ones((2,2),np.uint8)
thresh_dilated = cv2.dilate(thresh, kernel, iterations=3)

kernel = np.ones((2,2), np.uint8)  # Definisci il kernel di erosione
thresh_eroded = cv2.erode(thresh_dilated, kernel, iterations=2)  # Applica l'erosione

kernel = np.ones((2,2),np.uint8)
thresh_dilated = cv2.dilate(thresh, kernel, iterations=3)

kernel = np.ones((2,2), np.uint8)  # Definisci il kernel di erosione
thresh_eroded = cv2.erode(thresh_dilated, kernel, iterations=2)  # Applica l'erosione

kernel = np.ones((2,2),np.uint8)
thresh_dilated = cv2.dilate(thresh, kernel, iterations=2)

kernel = np.ones((2,2), np.uint8)  # Definisci il kernel di erosione
thresh_eroded = cv2.erode(thresh_dilated, kernel, iterations=4)  # Applica l'erosione

kernel = np.ones((2,2),np.uint8)
thresh_dilated = cv2.dilate(thresh, kernel, iterations=3)

kernel = np.ones((3,3), np.uint8)  # Definisci il kernel di erosione
thresh_eroded = cv2.erode(thresh_dilated, kernel, iterations=1)  # Applica l'erosione

kernel = np.ones((4,4),np.uint8)
thresh_dilated = cv2.dilate(thresh, kernel, iterations=1)

kernel = np.ones((3,3), np.uint8)  # Definisci il kernel di erosione
thresh_eroded = cv2.erode(thresh_dilated, kernel, iterations=1)  # Applica l'erosione

kernel = np.ones((2,2), np.uint8)  # Definisci il kernel di erosione
thresh_eroded = cv2.erode(thresh_eroded, kernel, iterations=1)  # Applica l'erosione

kernel = np.ones((4,4),np.uint8)
thresh_dilated = cv2.dilate(thresh, kernel, iterations=1)

kernel = np.ones((4,4), np.uint8)  # Definisci il kernel di erosione
thresh_eroded = cv2.erode(thresh_dilated, kernel, iterations=1)  # Applica l'erosione

kernel = np.ones((4,5),np.uint8)
thresh_dilated = cv2.dilate(thresh_eroded, kernel, iterations=1)

kernel = np.ones((2,2), np.uint8)  # Definisci il kernel di erosione
thresh_eroded = cv2.erode(thresh_dilated, kernel, iterations=3)  # Applica l'erosione

kernel = np.ones((4,4),np.uint8)
thresh_dilated = cv2.dilate(thresh_eroded, kernel, iterations=2)

kernel = np.ones((3,3), np.uint8)  # Definisci il kernel di erosione
thresh_eroded = cv2.erode(thresh_dilated, kernel, iterations=3)  # Applica l'erosione

kernel = np.ones((5,5),np.uint8)
thresh_dilated = cv2.dilate(thresh_eroded, kernel, iterations=2)

kernel = np.ones((4,4), np.uint8)  # Definisci il kernel di erosione
thresh_eroded = cv2.erode(thresh_dilated, kernel, iterations=3)  # Applica l'erosione

kernel = np.ones((4,4),np.uint8)
thresh_dilated = cv2.dilate(thresh_eroded, kernel, iterations=2)

# Trova i contorni nella nuova immagine binarizzata erosa
contours, _ = cv2.findContours(thresh_dilated , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crea un'immagine vuota per disegnare i contorni
contour_img = np.zeros_like(image, dtype=np.uint8)

# Crea un'immagine vuota per disegnare i contorni
contour_img = np.zeros_like(image, dtype=np.uint8)

# Disegna i contorni sull'immagine vuota
cv2.drawContours(contour_img, contours, -1, (255), thickness=cv2.FILLED)

# Visualizza l'immagine risultante
plt.imshow(contour_img, cmap='gray')
plt.axis('off')
plt.show()
plt.savefig('puzzlenoirblanc.jpg', image)
