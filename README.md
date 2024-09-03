# Clasificación de Dígitos MNIST con TensorFlow y Keras

Este proyecto implementa un modelo de red neuronal convolucional (CNN) para clasificar dígitos escritos a mano del conjunto de datos MNIST utilizando TensorFlow y Keras.

## Descripción

El proyecto incluye:
- Carga y preprocesamiento del conjunto de datos MNIST
- Implementación de una CNN para clasificación de imágenes
- Entrenamiento y evaluación del modelo
- Visualización de resultados y predicciones

## Requisitos

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

## Instalación

1. Clona este repositorio:
   ```
   git clone https://github.com/kovlak/mnist-digit-classification.git
   ```
2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso

1. Para entrenar el modelo y ver los resultados:
   ```
   python mnist_classification.py
   ```
2. Para visualizar muestras del conjunto de datos:
   ```
   python mnist_visualization.py
   ```

## Resultados

El modelo alcanzó una precisión del 98.67% en el conjunto de prueba. 

Las imágenes de las predicciones y las métricas de entrenamiento se guardan como 'mnist_predictions.png' y 'mnist_training_history.png' respectivamente.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de hacer un pull request.

## Licencia

Este proyecto está bajo la licencia MIT.# mnist-digit-classification
Proyecto de clasificación de dígitos MNIST usando TensorFlow y Keras
