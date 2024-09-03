import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Cargar el conjunto de datos MNIST
print("Cargando el conjunto de datos MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar los datos
print("Normalizando los datos...")
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape para añadir el canal de color (escala de grises)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# One-hot encoding de las etiquetas
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Definir el modelo
print("Definiendo el modelo CNN...")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar el modelo
print("Compilando el modelo...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
print("Entrenando el modelo...")
history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1, verbose=1)

# Evaluar el modelo
print("Evaluando el modelo...")
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Pérdida en el conjunto de prueba: {score[0]}')
print(f'Precisión en el conjunto de prueba: {score[1]}')

# Visualizar algunas predicciones
print("Generando predicciones...")
predictions = model.predict(x_test[:5])
fig, axes = plt.subplots(1, 5, figsize=(20,4))
for i, ax in enumerate(axes):
    ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(y_test[i])
    ax.set_title(f'Pred: {predicted_label}, Real: {true_label}')
    ax.axis('off')

# Guardar la figura con las predicciones
plt.savefig('mnist_predictions.png')
print("Imagen de predicciones guardada como 'mnist_predictions.png'")

# Graficar la precisión del entrenamiento y validación
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Graficar la pérdida de entrenamiento y validación
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Guardar la figura con las gráficas de entrenamiento
plt.savefig('mnist_training_history.png')
print("Gráficas de entrenamiento guardadas como 'mnist_training_history.png'")
