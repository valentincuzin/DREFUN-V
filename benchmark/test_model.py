import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Politique_CartPole(nn.Module):
    def __init__(self, dim_entree:int, dim_sortie:int , dim_cachee:int):
        super(Politique_CartPole, self).__init__()
        self.dim_entree = dim_entree
        self.dim_sortie = dim_sortie
        self.dim_cachee = dim_cachee
        self.fc1 = nn.Linear(self.dim_entree, self.dim_cachee)
        self.fc2 = nn.Linear(self.dim_cachee, self.dim_sortie)

    def forward(self, etat : np.ndarray):
        x = F.relu(self.fc1(etat))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    
    def action(self, etat : np.ndarray) -> tuple[int,torch.Tensor]:
        """
        Renvoi l'action a executer dans etat et la log proba de cette action
        """
        if isinstance(etat, np.ndarray):
            etat = torch.tensor(etat, dtype=torch.float).unsqueeze(0) 
        proba = self.forward(etat)
        m = Categorical(proba)
        action = m.sample()
        log_proba = m.log_prob(action)
        return action.item(), log_proba


def eval_politique(politique, env, nb_episodes=100, max_t=1000, seed=random.randint(0,5000), init_large=False) -> list:
    somme_rec = []
    total_rec = 0
    meilleurs_scores = 0
    for epi in range(1, nb_episodes+1):
        if init_large:
            state, _ = env.reset(seed=seed,options={'low':-0.2,'high':0.2})
        else:
            state, _ = env.reset(seed=seed)

        for i in range(1, max_t+1):
            action, _ = politique.action(state)
            next_observation, reward, terminated, truncated, info = env.step(action)
            total_rec += reward
            state = next_observation
            if terminated or truncated:
                epi += 1
                somme_rec.append(total_rec)
                if total_rec > meilleurs_scores:
                    meilleurs_scores = total_rec
                total_rec = 0
                break
    return somme_rec





def test_CartPole_model():
    env = gym.make("CartPole-v1")
    politique = Politique_CartPole(env.observation_space.shape[0], env.action_space.n, 12)
    politique.load_state_dict(torch.load("model/CartPole.pth"))
    politique.eval()  # Set the model to evaluation mode
    
    return eval_politique(politique, env, seed=5, nb_episodes=40)
    
    
    
if __name__ == "__main__":
    res = test_CartPole_model()
    print(res)