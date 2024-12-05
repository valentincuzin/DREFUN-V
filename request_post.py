import requests
import json

# Configuration de l'URL de l'API Ollama
OLLAMA_API_URL = "http://localhost:11434/api/generate"

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
        "prompt": prompt,
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
                return response.json().get('response', '')
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
    code_prompt = "Écris une fonction Python qui calcule la moyenne d'une liste de nombres"
    print("Génération de code Python :")
    code_response = generate_response(code_prompt)
    print(code_response)
    
    # Exemple de requête en mode streaming
    print("\nDémonstration du mode streaming :")
    streaming_prompt = "Explique l'algorithme de tri rapide en Python"
    for chunk in generate_response(streaming_prompt, stream=True):
        print(chunk, end='', flush=True)

if __name__ == "__main__":
    main()