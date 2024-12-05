import requests
import json
from typing import List, Dict, Union, Optional, Generator

OLLAMA_CHAT_API_URL = "http://localhost:11434/api/chat"

class OllamaChat:
    def __init__(
        self, 
        model: str = "qwen2.5-coder", 
        system_prompt: Optional[str] = None,
        options: Optional[Dict] = None
    ):
        """
        Initialize an advanced Ollama chat session with extended configuration.
        
        Args:
            model (str, optional): The name of the Ollama model.
            system_prompt (str, optional): Initial system message to set chat context.
            options (dict, optional): Advanced model generation parameters.
        """
        self.model = model
        self.messages: List[Dict[str, str]] = []
        self.options = options or {}
        
        if system_prompt:
            print(f"System: {system_prompt}")
            self.add_message(system_prompt, role="system")
    
    def add_message(
        self, 
        content: str, 
        role: str = "user", 
        **kwargs
    ) -> None:
        """
        Add a message to the chat history with optional metadata.
        
        Args:
            content (str): The message content
            role (str, optional): Message role (user/assistant/system)
        """
        message = {"role": role, "content": content, **kwargs}
        self.messages.append(message)
    
    def generate_response(
        self, 
        stream: bool = False, 
        additional_options: Optional[Dict] = None
    ) -> Union[str, Generator]:
        """
        Generate a response with advanced configuration options.
        
        Args:
            stream (bool, optional): Stream response in real-time
            additional_options (dict, optional): Temporary generation options
        
        Returns:
            Response as string or streaming generator
        """
        generation_options = {
            **self.options,
            **(additional_options or {})
        }
        
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": stream,
            "options": generation_options
        }
        
        try:
            response = requests.post(
                OLLAMA_CHAT_API_URL, 
                json=payload, 
                stream=stream
            )
            
            response.raise_for_status()  
            if not stream:
                full_response = response.json()
                assistant_response = full_response.get('message', {}).get('content', '')
                self.add_message(assistant_response, role="assistant")
                return assistant_response
            
            def stream_response():
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line.decode('utf-8'))
                            if 'message' in json_response:
                                chunk = json_response['message'].get('content', '')
                                full_response += chunk
                                yield chunk
                        except json.JSONDecodeError:
                            continue
                
                if full_response:
                    self.add_message(full_response, role="assistant")
            
            return stream_response()
        
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            return ""

def main():
    chat = OllamaChat(
        model="qwen2.5-coder",
        system_prompt="You are an expert in Reinforcement Learning. Don't explain, don't argue, give me only the reward methods for the RL problem. Give the code in ```python code blocks```",
        options={
        
        }
    )

    print("Programming conversation:\n")

    chat.add_message("I want a method for the reward methods in gymnasium, for the env acrobot")
    print("User: I want a method for the reward methods in gymnasium, for the env acrobot")

    print("\nAssistant: ", end='', flush=True)
    response = chat.generate_response(stream=True)
    for chunk in response:
        print(chunk, end='', flush=True)

if __name__ == "__main__":
    main()