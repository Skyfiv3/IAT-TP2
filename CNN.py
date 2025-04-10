import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import time


# === Callback personnalisé pour afficher les courbes de loss/accuracy ===
class PlotLossAccuracy(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = []
        self.train_loss, self.val_loss = [], []
        self.train_acc, self.val_acc = [], []

        self.fig, self.axs = plt.subplots(1, 2, figsize=(14, 5))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

        # Mise à jour des graphiques
        self.axs[0].cla()
        self.axs[1].cla()

        # Courbe de perte
        self.axs[0].plot(self.epochs, self.train_loss, label="Train Loss")
        self.axs[0].plot(self.epochs, self.val_loss, label="Validation Loss")
        self.axs[0].set_title("Model Loss")
        self.axs[0].set_xlabel("Epoch")
        self.axs[0].set_ylabel("Loss")
        self.axs[0].legend()
        self.axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        # Courbe de précision
        self.axs[1].plot(self.epochs, self.train_acc, label="Train Accuracy")
        self.axs[1].plot(self.epochs, self.val_acc, label="Validation Accuracy")
        self.axs[1].set_title("Model Accuracy")
        self.axs[1].set_xlabel("Epoch")
        self.axs[1].set_ylabel("Accuracy")
        self.axs[1].legend()
        self.axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()

    def on_train_end(self, logs=None):
        plt.tight_layout()
        plt.show()

# === Préparation des données ===
data_dir= 'archive/data'
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3  # Diviser 30% des données pour la validation
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='training'  # Utilisation de la partie entraînement (70%)
)
val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation',  # Utilisation de la partie validation (30%)
    shuffle=False
)

# === Modèle CNN ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(3, 3)),
    Dropout(0.3),
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(3, 3)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(3, 3)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Adapté à 1 classes
])

# === Compilation du modèle ===
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy','precision','recall']
)

model.summary()

# === Entraînement avec mesure du temps ===
plot_callback = PlotLossAccuracy()

start_time = time.time()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[plot_callback]
)

end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)

print(f"\n✅ Entraînement terminé en {int(minutes)} min {int(seconds)} sec.")

# === Évaluation & Matrice de confusion ===
val_generator.reset()
predictions = model.predict(val_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_generator.classes

# Affichage de la matrice de confusion
fig, ax = plt.subplots(figsize=(12, 8))
disp = ConfusionMatrixDisplay.from_predictions(
    true_classes,
    predicted_classes,
    normalize='true',
    cmap='Blues',
    values_format='.1%',
    display_labels=val_generator.class_indices.keys(), 
    ax=ax
)

plt.title("Matrice de confusion")
plt.xlabel("Label Prédit")
plt.ylabel("Label Réel")
plt.show()

# === Rapport de classification (optionnel mais utile) ===
class_labels = list(val_generator.class_indices.keys())
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
