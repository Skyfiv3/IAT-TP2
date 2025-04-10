import numpy as np
import matplotlib.pyplot as plt
import time
# --- 1. Créer le labyrinthe ---
# 0: vide, 1: mur, 2: départ, 3: but
maze = np.array([
    [0, 0, 0, 1, 3],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [2, 0, 0, 0, 0]])
# --- 2. Environnement RL ---
class MazeEnv:
    def __init__(self, maze):
        self.maze = maze.copy()
        self.start_pos = tuple(np.argwhere(maze == 2)[0])
        self.goal_pos = tuple(np.argwhere(maze == 3)[0])
        self.reset()
    
    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos
    
    def step(self, action):
        moves = [(-1,0), (1,0), (0,-1), (0,1)] # haut, bas, gauche, droite
        move = moves[action]
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])
        if not (0 <= new_pos[0] < self.maze.shape[0] and 0 <= new_pos[1] <self.maze.shape[1]):
            return self.agent_pos, -1, False
        if self.maze[new_pos] == 1:
            return self.agent_pos, -1, False
        self.agent_pos = new_pos
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 10, True
        return self.agent_pos, -0.1, False
    
    
# --- 3. Q-learning setup ---
env = MazeEnv(maze)
q_table = np.zeros((5, 5, 4)) # états (5x5), 4 actions
alpha = 0.1
gamma = 0.9
epsilon = 1
episodes = 100




# --- 4. Entraînement ---
rewards_per_episode = []  # Liste pour stocker les récompenses totales

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False
    print(f"\nÉpisode {ep+1}")
    while not done and steps < 100:
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_table[state[0], state[1]])
            
        next_state, reward, done = env.step(action)
        
        q_old = q_table[state[0], state[1], action]
        q_next = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action] = q_old + alpha * (reward + gamma * q_next - q_old)
        
        state = next_state
        total_reward += reward
        steps += 1
    rewards_per_episode.append(total_reward)  # Enregistrer la récompense totale
    print(f" Terminé en {steps} étapes | Reward total = {round(total_reward, 2)}")

# --- 5. Politique finale ---
print("\n Politique finale (meilleure action par case) :")
action_map = ['↑', '↓', '←', '→']
policy = np.full((5, 5), ' ')
for i in range(5):
    for j in range(5):
        if maze[i, j] == 1:
            policy[i, j] = '█'
        elif maze[i, j] == 3:
            policy[i, j] = '🎯'
        elif maze[i, j] == 2:
            policy[i, j] = 'S'
        else:
            best_a = np.argmax(q_table[i, j])
            policy[i, j] = action_map[best_a]
print(policy)


# --- 6. Tracer la courbe reward/épisode ---
plt.figure(figsize=(10, 6))

# Ajouter les hyperparamètres dans le titre
plt.plot(range(1, episodes + 1), rewards_per_episode, marker='o', label='Reward par épisode')
plt.title(f"Évolution des récompenses par épisode\n"
          f"(alpha={alpha}, gamma={gamma}, epsilon={epsilon}, episodes={episodes})")
plt.xlabel('Épisode')
plt.ylabel('Reward total')
plt.grid()
plt.legend()

# Afficher la reward finale sur la courbe
final_reward = rewards_per_episode[-1]
plt.annotate(f"Reward finale: {final_reward:.2f}",
             xy=(episodes, final_reward),
             xytext=(episodes - 2, final_reward + 5),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=10)

plt.show()