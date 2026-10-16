import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración de parámetros y reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

print(f"Versión de TensorFlow en uso: {tf._version_}")

# 2. Carga y preparación del conjunto de datos (MNIST)
mnist = tf.keras.datasets.mnist
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

# Normalización de los tensores de entrada al rango [0.0, 1.0]
X_train = X_train_raw / 255.0
X_test = X_test_raw / 255.0

print(f"Forma de datos de entrenamiento: {X_train.shape}")
print(f"Forma de datos de prueba: {X_test.shape}")

# 3. Construcción de la Arquitectura Multilayer Perceptron (MLP)
def construir_mlp():
    model = models.Sequential([
        # Capa de entrada explícita recomendada en Keras moderno
        layers.Input(shape=(28, 28), name="Capa_Entrada"),
        
        # Capa de aplanamiento: convierte matrices (28, 28) en vectores de 784 entradas
        layers.Flatten(name="Capa_Aplanamiento"),
        
        # Primera Capa Oculta Densa con activación ReLU
        layers.Dense(128, activation='relu', name="Capa_Oculta_1"),
        
        # Segunda Capa Oculta Densa
        layers.Dense(64, activation='relu', name="Capa_Oculta_2"),
        
        # Capa de Salida Multiclase con activación Softmax (10 dígitos)
        layers.Dense(10, activation='softmax', name="Capa_Salida_Softmax")
    ], name="MLP_UNINTEP_RA2")
    
    return model

modelo_mlp = construir_mlp()

# Muestra el resumen estructural de parámetros entrenables
modelo_mlp.summary()

# 4. Compilación del Modelo
TASA_DE_APRENDIZAJE = 0.001

optimizador_adam = optimizers.Adam(learning_rate=TASA_DE_APRENDIZAJE)

modelo_mlp.compile(
    optimizer=optimizador_adam,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Proceso de Entrenamiento (Forward - Loss - Backpropagation)
TAMANO_LOTE = 64
EPOCAS = 10

print("\nIniciando entrenamiento del modelo...")
historial = modelo_mlp.fit(
    X_train, y_train_raw,
    epochs=EPOCAS,
    batch_size=TAMANO_LOTE,
    validation_split=0.2,
    verbose=1
)

# 6. Evaluación en datos nunca antes vistos (Prueba)
loss_test, accuracy_test = modelo_mlp.evaluate(X_test, y_test_raw, verbose=0)
print(f"\n==========================================")
print(f"Pérdida en Test (Loss): {loss_test:.4f}")
print(f"Precisión en Test (Accuracy): {accuracy_test * 100:.2f}%")
print(f"==========================================")
