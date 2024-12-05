import requests
import json

# Configuration de l'URL de l'API Ollama
OLLAMA_API_URL = "http://localhost:11434/api/chat"

def generate_response(prompt, model="qwen2.5-coder", stream=False):
    """
    Génère une réponse en utilisant le modèle Ollama spécifié.
    
    Args:
        prompt (str): Le texte de requête à envoyer au modèle
        model (str, optional): Le nom du modèle Ollama. Défaut à "qwen2.5-coder"
        stream (bool, optional): Si True, retourne la réponse en streaming. Défaut à False
    
    Returns:
        str or generator: La réponse générée par le modèle
    """
    payload = {
        "model": model,
        "messages": prompt,
        "stream": stream
    }
    
    try:
        # Envoi de la requête à l'API Ollama
        response = requests.post(
            OLLAMA_API_URL, 
            json=payload, 
            stream=stream
        )
        
        # Gestion du mode non-streaming
        if not stream:
            if response.status_code == 200:
                return response.json().get('message', '')
            else:
                raise Exception(f"Erreur API: {response.status_code}")
        
        # Gestion du mode streaming
        def stream_response():
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        json_response = json.loads(decoded_line)
                        if 'response' in json_response:
                            yield json_response['response']
                    except json.JSONDecodeError:
                        continue
        
        return stream_response()
    
    except requests.exceptions.RequestException as e:
        print(f"Erreur de connexion: {e}")
        return ""

# Exemples d'utilisation
def main():
    # Exemple de génération de code Python
    code_prompt = [
    {
      "role": "user",
      "content": "In cartpole env, make the reward function with an observation space:.\n"
                +"The cart x-position (index 0) can be take values between (-4.8, 4.8), but the episode terminates if the cart leaves the (-2.4, 2.4) range.\n"
                +"The pole angle can be observed between (-.418, .418) radians (or ±24°), but the episode terminates if the pole angle is not in the range (-.2095, .2095) (or ±12°)\n"
    }
  ]
    print("Génération de code Python :")
    code_response = generate_response(code_prompt)
    print(code_response)

if __name__ == "__main__":
    main()