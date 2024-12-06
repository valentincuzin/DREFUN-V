import gymnasium as gym
import numpy as np
from typing import List, Dict, Callable, Optional

from OllamaChat import OllamaChat

class DREFUN:
    def __init__(
            self, 
            model : str ="qwen2.5-coder",
            learning_method: str = "REINFORCE"
        ):
        
        """
        Initialize DREFUN architecture for dynamic reward function generation
            Args:
                model (str): Language model for reward generation
                learning_method (str): Reinforcement learning method
        """
        self.llm = OllamaChat(
        model="qwen2.5-coder",
        system_prompt = """
        You are an expert in Reinforcement Learning specialized in designing reward functions. 
        Strict criteria:
        - Provide ONLY the reward function code
        - Use Python format
        - Briefly comment on the function's logic
        - Give no additional explanations
        - Focus on the Gymnasium Acrobot environment
        - STOP immediately after closing the ``` code block
        """,
    
        options = {
        "temperature": 0.2,
        "max_tokens": 300,
        
        }
    )
        self.learning_methods = {
            "REINFORCE": self._reinforce_learning,
            "DirectSearch": self._direct_search
        }

        self.current_learning_method = self.learning_methods[learning_method]

        self.reward_functions: List[Callable] = []
        self.policy_performances: List[Dict] = []
        self.benchmark_environments: List[gym.Env] = []

    def generate_reward_function(
        self, 
        task_description: str, 
        environment_type: str
    ) -> Callable:
        """
        Generate reward function using LLM
        
        Args:
            task_description (str): Detailed description of the task
            environment_type (str): Type of environment (2D/3D, robotics, etc.)
        
        Returns:
            Callable: Generated reward function
        """
        prompt = f"""
        Generate a reward function for a {environment_type} environment.
        Task Description: {task_description}
        
        Requirements:
        - Provide clean, efficient implementation
        """
        
        self.llm.add_message(prompt)
        response = self.llm.generate_response()

        # print("LLM Response:", response)
        
        reward_func = self._compile_reward_function(response)
        self.reward_functions.append(reward_func)
        
        return reward_func
    

    def _compile_reward_function(self, response: str) -> Callable:
        """
        Compile the reward function from the LLM response.
        TODO BUG avec ``` a la fin jsp pk mais on va trouver les gars !
        
        Args:
            response (str): LLM generated reward function.
        
        Returns:
            Callable: Compiled reward function.
        """
        cleaned_response = response.strip("```").replace("python", "").strip()

        if "def " not in cleaned_response:
            raise ValueError("La réponse ne contient pas de définition de fonction valide.")

        print("Code nettoyé pour compilation :\n", cleaned_response)


        exec_globals = {}
        try:
            exec(cleaned_response, exec_globals)
        except SyntaxError as e:
            raise ValueError(f"Erreur de syntaxe dans le code généré : {e}")

        reward_function_name = cleaned_response.split('(')[0].split()[-1]
        reward_function = exec_globals.get(reward_function_name)
        if not callable(reward_function):
            raise ValueError("La fonction reward n'a pas été trouvée ou n'est pas valide.")
        
        return reward_function
    
    def test_reward_function(self, reward_function: Callable, *args, **kwargs):
        """
        Test the compiled reward function with example inputs.
        
        Args:
            reward_function (Callable): The reward function to test.
            *args: Positional arguments for the reward function.
            **kwargs: Keyword arguments for the reward function.
        """
        try:
            reward = reward_function(*args, **kwargs)
            print(f"Reward function output: {reward}")
        except Exception as e:
            raise RuntimeError(f"Error during reward function execution: {e}")
    
    def _reinforce_learning(self):
        """
        TODO Implement REINFORCE learning method
        """
        pass

    def _direct_search(self):
        """
        TODO Implement Direct Search learning method
        """
        pass


    def self_refine_reward(
        self, 
        current_reward_func: Callable, 
        performance_metrics: Dict
    ) -> Callable:
        """
        Self-refinement of reward function based on performance
        
        Args:
            current_reward_func (Callable): Current reward function
            performance_metrics (Dict): Performance evaluation metrics
        
        Returns:
            Callable: Refined reward function
        """
        refinement_prompt = f"""
        Current reward function performance:
        {performance_metrics}
        
        Suggest improvements to the reward function to:
        - Increase success rate
        - Optimize reward signal
        - Maintain task objectives
        """
        
        self.llm.add_message(refinement_prompt)
        refined_response = self.llm.generate_response()
        
        return self._compile_reward_function(refined_response)
    
    def evaluate_policy(
        self, 
        env: gym.Env, 
        reward_func: Callable, 
        num_episodes: int = 10
    ) -> Dict:
        """
        Evaluate policy performance for a given reward function
        
        Args:
            env (gym.Env): Gymnasium environment
            reward_func (Callable): Reward function to evaluate
            num_episodes (int): Number of evaluation episodes
        
        Returns:
            Dict: Performance metrics
        """
        performance_metrics = {
            "total_rewards": [],
            "episode_lengths": [],
            "success_rate": 0.0
        }
        
        # TODO
        
        return performance_metrics
    
    def run_benchmark(
        self, 
        environments: List[gym.Env], 
        num_iterations: int = 5
    ):
        """
        Run benchmark across multiple environments
        
        Args:
            environments (List[gym.Env]): List of environments to test
            num_iterations (int): Number of iterations per environment
        """
        benchmark_results = {}
        
        for env in environments:
            env_results = []
            
            for _ in range(num_iterations):
                reward_func = self.generate_reward_function(
                    task_description=env.unwrapped.spec.id,
                    environment_type=""
                )
                
                # Evaluate and refine
                performance = self.evaluate_policy(env, reward_func)
                # performance = 0
                refined_reward = self.self_refine_reward(
                    reward_func, 
                    performance
                )
                
                env_results.append({
                    "initial_performance": performance,
                    "refined_performance": self.evaluate_policy(
                        env, refined_reward
                    )
                })
            
            benchmark_results[env.unwrapped.spec.id] = env_results
        
        return benchmark_results



def main():

    benchmark_envs = [
    gym.make('CartPole-v1')
    ]
    
    
    drefun = DREFUN()

    reward_func = drefun.generate_reward_function(
        task_description="Balance a pole on a cart",
        environment_type="2D"
    )

    print(reward_func)

    drefun.test_reward_function(
        reward_func, 
        observation=[0.0, 0.1, -0.2, 0.3],
    )










    results = drefun.run_benchmark(
        environments=benchmark_envs,
        num_iterations=1
    )

    for env_name, env_results in results.items():
        print(f"\nEnvironment: {env_name}")
        for i, iteration in enumerate(env_results, 1):
            print(f"Iteration {i}:")
            print("Initial Performance:", iteration['initial_performance'])
            print("Refined Performance:", iteration['refined_performance'])
            print("-" * 50)

if __name__ == "__main__":
    main()