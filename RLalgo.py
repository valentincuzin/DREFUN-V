from utils import eval_politique

import numpy as np
import random
import torch
import gymnasium as gym


class PolitiqueDirectSearch:
    def __init__(
            self, 
            dim_entree:int, 
            dim_sortie:int, 
            det = True
        ):
        self.dim_entree = dim_entree
        self.dim_sortie = dim_sortie
        self.det = det
        # Matrice entre * sortie
        self.poids = np.random.rand(dim_entree, dim_sortie)

    def output(self, etat : np.ndarray) -> int:
        """Calcul de la sortie de la politique
        - si déterministe : argmax
        - si stochastique : probabilité de chaque action
        """
        # Nous utilisons une fonction d'activation soft max pour que les poids soient à la même échelle
        prob = torch.nn.functional.softmax(torch.tensor(etat.dot(self.poids)), dim=0)
        if self.det:
            return torch.argmax(prob).item()
        else:
            return torch.Categorical(probs=prob).sample().item()

    def set_poids(self, poids: np.ndarray):
        self.poids = poids

    def get_poids(self) -> np.ndarray:
        return self.poids

    def save(self, file):
        f = open(file, "w")
        f.write(self.poids + ";" + self.det)

    def load(self, file):
        f = open(file, "r")
        param = f.read().split(";")
        self.poids = param[0]
        self.det = param[1]

    def rollout(self, env, reward_func, max_t=1000) -> int:
        """
        execute un episode sur l'environnement env avec la politique et renvoie la somme des recompenses obtenues sur l'épisode
        """
        total_rec = 0
        state, _ = env.reset(seed=random.randint(0, 5000))
        for _ in range(1, max_t + 1):
            action = self.output(state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            if reward_func is not None:
                # print("next_observation", next_observation, "action", action, "reward", reward, "reward func", reward_func(next_observation, action))
                reward = reward_func(next_observation, action)
            total_rec += reward
            state = next_observation
            if terminated or truncated:
                return total_rec
        return total_rec

    def train(
        self, env, reward_func=None, nb_episodes=5000, max_t=1000
    ) -> tuple[list, np.ndarray]:
        bruit_std = 1e-2
        meilleur_perf = 0
        meilleur_poid = self.get_poids()
        pref_by_episode = list()
        nb_500_affile = 0
        for i_episode in range(1, nb_episodes + 1):
            perf = self.rollout(env, reward_func, max_t)
            pref_by_episode.append(perf)

            if perf == 500:
                nb_500_affile += 1
            else:
                nb_500_affile = 0
            if nb_500_affile == 10:
                return pref_by_episode, meilleur_poid

            if perf >= meilleur_perf:
                meilleur_perf = perf
                meilleur_poid = self.get_poids()
                # reduction de la variance du bruit
                bruit_std = max(1e-3, bruit_std / 2)
            else:
                # augmentation de la variance du bruit
                bruit_std = min(2, bruit_std * 2)
             # On calcule le bruit en fonction de la variance
            bruit = np.random.normal(0, bruit_std, self.dim_entree * self.dim_sortie)
            # Reshape le bruit pour qu'il ait la même taille que les poids
            bruit = bruit.reshape(self.dim_entree, self.dim_sortie)
            # if i_episode % 20 == 0:
            #     print(f"Episode {i_episode}, perf = {perf}, best perf = {meilleur_perf}, bruit = {bruit_std}")
            # On ajoute le bruit aux poids
            self.set_poids(self.get_poids() + bruit)
        return pref_by_episode, meilleur_poid


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    policy = PolitiqueDirectSearch(4, 2)
    policy.train(env)
    print(eval_politique(policy, env))
