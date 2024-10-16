# Detección de motosierras

## Base de datos utillizada
La base de datos utilizada se encuentra disponible en el [siguiente link](https://drive.google.com/file/d/17Pi7ygxCVGFVXwQwu6RJXOMhb76KE0v9/view?usp=sharing). La carpeta se estructura de la siguiente manera:

-dataset <br/>
--Bosques <-- Samples de bosques obtenidos de bibliotecas en linea <br/>
--Motosierras <-- Samples de motosierras obtenidas de blibliotecas en linea <br/>
--Sonidos_grabados <-- Sonidos de motosierras y ruido de fondo obtenido en campo <br/>

Si se desea recrear el proceso de entrenaimento debe descomprimirse el archivo del link en la carpeta del repositorio. 

## Detalles del modelo
![imagen](https://github.com/user-attachments/assets/a4a02b49-59ee-49db-a741-fd70a4f800f7)

El modelo toma archivos de audio muestreados a 16kHz con una profundidad de 16bits. El modelo Yamnet realiza la extraccion de espectrogramas y posteriormente de embeddings con una ventana de 0.96 segundos. Posterior a eso el modelo desarrollado realiza la clasificacion.

## Inferencia
Para hacer inferencias con el modelo desarrollado se debe ejecutar main_lite.py en la carpeta "inferencia". Dicho codigo esta preparado para capturar audio en tiempo real y arrojar resultados.

## Resultados
### Aumento de datos
Las tecnicas de aumento de datos evaluadas fueron:
-	Desplazamiento tonal (PS)
-	Agregado de ruido de fondo (BN)
-	Máscaras en frecuencia (FM)
-	Máscaras de tiempo (TM)
-	Desplazamiento temporal + Ruido de fondo (PS + BN) 
-	Enmascaramiento en frecuencia y tiempo (FM + TM) 
-	Todas las técnicas en simultaneo (ALL) 
-	Ninguna técnica (S/A)

![det-roc](https://github.com/user-attachments/assets/353c724d-7402-4a9e-a2ed-0ab7da046d7b)

### Evaluación final en campo
La imagen siguiente representa un expectrograma del audio capturado durante la prueba final. Las areas resaltadas en celeste son las detecciones realizadas por el algoritmo.

![deteccion](https://github.com/user-attachments/assets/06cc53c0-3661-4b85-83b6-5a195e368296)

