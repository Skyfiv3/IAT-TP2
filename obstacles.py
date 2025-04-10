import os
import cv2
import numpy as np
import random
import time
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Charger le dataset depuis Kaggle
def load_dataset(folder_path, label):
    dataset = []
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convertir en niveaux de gris
        img = cv2.resize(img, (64, 64))  # Redimensionner
        features = extract_hog_features(img)  # Extraire les caractéristiques HOG
        dataset.append({'features': features, 'label': label})

    return dataset


# Extraction des descripteurs HOG
def extract_hog_features(image):
    features, _ = hog(image, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), orientations=9, visualize=True, block_norm='L2-Hys')
    return features


# Ajouter du bruit aux données
def add_noise(dataset, noise_factor=0.2):
    noisy_dataset = []
    for data in dataset:
        noisy_features = np.array(data['features']) + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=len(data['features']))
        noisy_features = np.clip(noisy_features, 0., 1.)
        noisy_dataset.append({'features': noisy_features, 'label': data['label']})
    return noisy_dataset


# Charger les images des deux classes
vehicle_images = load_dataset("./data/vehicles", label=1)
non_vehicle_images = load_dataset("./data/non-vehicles", label=0)
dataset = vehicle_images + non_vehicle_images

# Diviser le dataset en training (70%) et test (30%)
train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)

# Normaliser les caractéristiques pour le training et le test
scaler = StandardScaler()
features_matrix = np.array([data['features'] for data in train_data])
scaled_features = scaler.fit_transform(features_matrix)

# Mettre à jour le dataset d'entraînement avec les caractéristiques normalisées
for i, data in enumerate(train_data):
    train_data[i]['features'] = tuple(scaled_features[i])

# Normaliser les données de test avec le même scaler
test_features_matrix = np.array([data['features'] for data in test_data])
scaled_test_features = scaler.transform(test_features_matrix)

for i, data in enumerate(test_data):
    test_data[i]['features'] = tuple(scaled_test_features[i])


class Environment:
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self.dataset[self.current_index]['features']

    def step(self, action):
        reward = 1 if action == self.dataset[self.current_index]['label'] else -1
        self.current_index += 1
        done = self.current_index >= len(self.dataset)
        next_state = self.dataset[self.current_index]['features'] if not done else None
        return next_state, reward, done


# Agent Q-Learning
class QLearningAgent:
    def __init__(self, action_space):
        self.q_table = {}
        self.action_space = action_space
        self.learning_rate = 0.3
        self.gamma = 0.7

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
        return np.argmax(self.q_table[state]) if random.random() > 0.1 else random.randint(0, self.action_space - 1)

    def update_q_value(self, state, action, reward, next_state):
        if next_state is None:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table.get(next_state, np.zeros(self.action_space)))

        # Mise à jour de la Q-table
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])


# Function to plot accuracy per episode
def plot_accuracy(training_accuracies, validation_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(training_accuracies, label="Training Accuracy", color="blue")
    plt.plot(validation_accuracies, label="Validation Accuracy", color="orange")
    plt.title('Accuracy per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.show()


# Simulation avec les données d'entraînement
env = Environment(train_data)
agent = QLearningAgent(action_space=2)
episode_rewards = []
training_accuracies = []
validation_accuracies = []

# Variables pour calculer l'accuracy par échantillon
correct_predictions = 0
total_predictions = 0

# Mesurer le temps d'entraînement
start_time = time.time()

for episode in range(10):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_value(state, action, reward, next_state)
        state = next_state
        total_reward += reward

        # Vérifier si l'action correspond au label réel
        if action == env.dataset[env.current_index - 1]['label']:
            correct_predictions += 1
        total_predictions += 1

    episode_rewards.append(total_reward)

    # Calculer l'accuracy cumulative sur le training
    training_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    training_accuracies.append(training_accuracy)

    # Évaluer sur le set de validation
    test_env = Environment(test_data)
    correct_predictions_val = 0
    total_predictions_val = 0

    state = test_env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = test_env.step(action)
        if action == test_env.dataset[test_env.current_index - 1]['label']:
            correct_predictions_val += 1
        total_predictions_val += 1
        state = next_state

    validation_accuracy = (correct_predictions_val / total_predictions_val) * 100 if total_predictions_val > 0 else 0
    validation_accuracies.append(validation_accuracy)

    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Training Accuracy: {training_accuracy:.2f}%, Validation Accuracy: {validation_accuracy:.2f}%")

end_time = time.time()
print(f"Temps d'entraînement RL : {end_time - start_time:.2f} secondes")

# Plot the results
plot_accuracy(training_accuracies, validation_accuracies)