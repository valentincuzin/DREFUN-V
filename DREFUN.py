import gymnasium as gym
import numpy as np
from typing import List, Dict, Callable, Optional

from OllamaChat import OllamaChat
from logging import getLogger

logger = getLogger('DREFUN')

class DREFUN:
    def __init__(
        self,
        learning_method: Callable,
        env,
        model: str = "qwen2.5-coder",
        options: dict = {}
    ):
        """
        Initialize DREFUN architecture for dynamic reward function generation
            Args:
                model (str): Language model for reward generation
                learning_method (str): Reinforcement learning method
        """
        self.llm = OllamaChat(
            model=model,
            system_prompt="""
        You are an expert in Reinforcement Learning specialized in designing reward functions. 
        Strict criteria:
        - Provide dependancy if needed
        - Provide ONLY the reward function code
        - Use Python format
        - Briefly comment on the function's logic
        - Give no additional explanations
        - Focus on the Gymnasium environment 
        - Take into the action space
        - STOP immediately after closing the ``` code block
        """,
            options=options,
        )

        self.env = env
        self.learning_method = learning_method

        self.reward_functions: List[Callable] = []
        self.policy_performances: List[Dict] = []

    def generate_reward_function(self, task_description: str) -> Callable:
        """
        Generate reward function using LLM

        Args:
            task_description (str): Detailed description of the task
            environment_type (str): Type of environment (2D/3D, robotics, etc.)

        Returns:
            Callable: Generated reward function
        """
        prompt = f"""
        Generate a reward function for a {self.env.spec.name} environment.
        Task Description: {task_description}
        
        Requirements:
        - Provide clean, efficient implementation
        - Take into the account action space
        """  # TODO better prompt with completion of the function

        self.llm.add_message(prompt)
        response = self.llm.generate_response()  # TODO generate 2 responses        
        response = self._get_code(response)
        reward_func = self._get_runnable_function(response)
        self.reward_functions.append(reward_func)

        return reward_func

    def _get_code(self, response: str) -> str:
        cleaned_response = response.strip("```").replace("python", "").strip()
        if "def " not in cleaned_response:
            raise ValueError(
                "La réponse ne contient pas de définition de fonction valide."
            )
        logger.info("Code nettoyé pour compilation :\n" + cleaned_response)
        return cleaned_response

    def _get_runnable_function(self, response: str, error: str=None) -> Callable:
        if error is not None:
            self.llm.add_message(error)
            response = self.llm.generate_response()
            response = self._get_code(response)
        try:
            reward_func = self._compile_reward_function(response)
            state, _ = self.env.reset()
            action = self.learning_method.output(state)
            self._test_reward_function(
                reward_func,
                observation=state,
                action=action,
            )
        except SyntaxError as e:
            logger.warning(f"Error syntax {e}")
            return self._get_runnable_function(response, str(e))
        except RuntimeError as e:
            logger.warning(f"Error execution {e}")
            return self._get_runnable_function(response, str(e))
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

        exec_globals = {}
        exec_globals['np'] = np
        try:
            exec(response, exec_globals)
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in the generated code : {e}")

        reward_function_name = response.split("(")[0].split()[
            -1
        ]  # récup le nom de la fonction
        reward_function = exec_globals.get(reward_function_name)

        return reward_function

    def _test_reward_function(self, reward_function: Callable, *args, **kwargs):
        """
        Test the compiled reward function with example inputs.

        Args:
            reward_function (Callable): The reward function to test.
            *args: Positional arguments for the reward function.
            **kwargs: Keyword arguments for the reward function.
        """
        try:
            reward = reward_function(*args, **kwargs)
            logger.debug(f"Reward function output: {reward}")
        except (
            Exception
        ) as e:
            raise RuntimeError(f"Error during reward function execution: {e}")

    def self_refine_reward(
        self, current_reward_func: Callable, performance_metrics: Dict
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

    def evaluate_policy(  # penser aux métrics objective propre à l'environnment
        self, num_episodes: int = 100, visual: bool = False
    ) -> Dict:
        """
        Evaluate policy performance for a given reward function

        Args:
            num_episodes (int): Number of evaluation episodes

        Returns:
            Dict: Performance metrics
        """
        performance_metrics = {
            "total_rewards": [],
            "episode_lengths": [],
            "success_rate": 0.0,
        }
        # Faire l'entrainement de la politique pour voir la vitesse d'apprentissage, avec success_rate
        raw_weight, raw_perfs, raw_sr = self.learning_method.train(self.env)
        weight, perfs, sr = self.learning_method.train(self.env, self.reward_functions[-1])
        # Faire le test de la politique optimale apprise, puis 
        return performance_metrics

    def run_benchmark(
        self, environments: List[gym.Env], num_iterations: int = 5
    ):  # TODO pas dans cette class
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
                    task_description=env.unwrapped.spec.id, environment_type=""
                )

                # Evaluate and refine
                performance = self.evaluate_policy(env, reward_func)
                # performance = 0
                refined_reward = self.self_refine_reward(reward_func, performance)

                env_results.append(
                    {
                        "initial_performance": performance,
                        "refined_performance": self.evaluate_policy(
                            env, refined_reward
                        ),
                    }
                )

            benchmark_results[env.unwrapped.spec.id] = env_results

        return benchmark_results
