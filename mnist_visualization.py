import tensorflow as tf
import matplotlib.pyplot as plt

# Cargar el conjunto de datos MNIST
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

# Crear una figura con subplots
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()  # Aplanar el array de ejes

for i in range(10):
    # Seleccionar una imagen aleatoria de cada clase
    idx = (y_train == i).nonzero()[0][0]
    
    # Mostrar la imagen
    axes[i].imshow(x_train[idx], cmap='gray')
    axes[i].set_title(f'Dígito: {i}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('mnist_samples.png')
print("Imagen de muestras guardada como 'mnist_samples.png'")
