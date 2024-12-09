import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


def init_seed(seedval):
    torch.manual_seed(seedval)
    np.random.seed(seedval)
    random.seed(seedval)
    
    
    
import torch.nn as nn
import torch.nn.functional as F


class Politique(nn.Module):
    def __init__(self, dim_entree:int, dim_sortie:int, couche_cachee:list):
        """
        dim_entree: Dimension de l'entrée (taille de l'état)
        dim_sortie: Dimension de la sortie (nombre d'actions)
        couche_cachee: Liste des tailles des couches cachées (exemple [64, 32])
        """
        super(Politique, self).__init__()
        self.dim_entree = dim_entree
        self.dim_sortie = dim_sortie
        
        # Créer dynamiquement les couches cachées
        self.couches_cachees = []
        input_size = self.dim_entree
        for hidden_size in couche_cachee:
            self.couches_cachees.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.couches_cachees = nn.ModuleList(self.couches_cachees)
        
        # La dernière couche qui produit la sortie
        self.fc_out = nn.Linear(input_size, self.dim_sortie)

    def forward(self, etat: torch.Tensor) -> torch.Tensor:
        """
        Etat: un tenseur d'état (entrée)
        """
        x = etat
        for layer in self.couches_cachees:
            x = F.relu(layer(x))
        x = self.fc_out(x)
        return F.softmax(x, dim=1)  # Softmax pour obtenir une distribution de probabilité

    def action(self, etat: np.ndarray) -> tuple[int, torch.Tensor]:
        """
        Renvoi l'action à exécuter dans l'état et la log-proba de cette action.
        """
        if isinstance(etat, np.ndarray):
            etat = torch.tensor(etat, dtype=torch.float).unsqueeze(0)  # Ajouter une dimension pour le batch
        proba = self.forward(etat)
        m = Categorical(proba)
        action = m.sample()
        log_proba = m.log_prob(action)
        return action.item(), log_proba

        
    
    
    

def trajectoire(env,politique, max_t:int =500)->  tuple[list,list]:
    """
        max_t: nombre max de pas de la trajectoire
    """
    etat, _ = env.reset(seed=random.randint(0,5000))
    recompenses = []
    log_probas = []
    for t in range(max_t):
        action, log_proba = politique.action(etat)
        etat_suivant, recompense, fini, _,_ = env.step(action)
        recompenses.append(recompense)
        log_probas.append(log_proba)
        if fini:
            break
        etat = etat_suivant

    return recompenses, log_probas



def loss_reinforce2(log_probs:list, retours_cumules:list):
    """
        log_probs: liste des log proba des actions prises à chaque pas d'une trajectoire 
        retours_cumules : liste des retours cumulés à chaque pas de la trajectoire
        renvoi: loss = - sum_t logproba(a_t) * retour_cumule_t
    """
    loss = []
    for log_prob,r_cumul in zip(log_probs,retours_cumules):
        loss.append(-log_prob * r_cumul) 
    return torch.cat(loss).sum()


def retours_cumules(recompenses : list, gamma: float =0.99):
    """
       recompenses: liste des récompenses reçues à chaque pas de la trajectoire
       renvoi: retour cumulé à chaque pas de la trajectoire
    """
    retour_cum = []
    for i in range(len(recompenses)):
        rec_cum_t = 0
        for rec in recompenses[i:]:
            rec_cum_t += gamma*rec
        retour_cum.append(rec_cum_t)
    return retour_cum


def reinforce2(env,politique, nb_episodes=2000, gamma=0.99,  max_t=500, model_name="temp") -> list:
    recompenses = []
    optimizer = optim.Adam(politique.parameters(), lr=1e-3)
    a_la_suite = 0
    sauvegarde = False
    for ep in range(nb_episodes):
        optimizer.zero_grad()
        recompense_ep, log_proba_ep = trajectoire(env, politique,max_t)
        retours_cum = retours_cumules(recompense_ep)
        recompenses.append(sum(recompense_ep))
        loss = loss_reinforce2(log_proba_ep, retours_cum)
        loss.backward()
        optimizer.step()
        if recompenses[-1] >= 200:
            a_la_suite += 1
        else:
            a_la_suite = 0
        
        if a_la_suite == 50 and not sauvegarde:
            #sauvegarde du modèle dans le dossier model
            torch.save(politique.state_dict(), "model/"+model_name)
            sauvegarde = True
            
        if ep % 100 == 0:
            print(f"ep: {ep}, Rec: {sum(recompense_ep)}")
    return recompenses




if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    politique = Politique(
        dim_entree=env.observation_space.shape[0],  # 8 dimensions de l'état
        dim_sortie=env.action_space.n,  # 4 actions possibles
        couche_cachee=[ 128, 256, 64, 32]
    )
    recompenses = reinforce2(env, politique, nb_episodes=5000, gamma=0.99, max_t=1000, model_name="LunarLander.pth")
