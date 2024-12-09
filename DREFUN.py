from logging import getLogger
from typing import Callable, Dict, Generator, List

import gymnasium as gym
import numpy as np

from OllamaChat import OllamaChat
from logging import getLogger
import logging


class DREFUN:
    def __init__(
        self,
        learning_method: Callable,
        env,
        model: str = "qwen2.5-coder",
        options: dict = {},
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
        - Complete ONLY the reward function code
        - Use Python format
        - Give no additional explanations
        - Focus on the Gymnasium environment 
        - Take into the observation of the state, the terminated and truncated boolean
        - STOP immediately your completion after the last return
        """,
            options=options,
        )

        self.env = env
        self.learning_method = learning_method

        self.reward_functions: List[Callable] = []
        self.policy_performances: List[Dict] = []

        self.logger = getLogger("DREFUN")
        if logging.getLevelName(self.logger.level) == "DEBUG":
            self.isStream = True
        else:
            self.isStream = False

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
        Complete the reward function for a {self.env.spec.name} environment.
        Task Description: {task_description}

        complete this sentence:
        def reward_func(observations:np.ndarray, terminated: bool, truncated: bool) -> float:
            \"\"\"Reward function for {self.env.spec.name}

            Args:
                observations (np.ndarray): observation on the current state
                terminated (bool): episode is terminated due a failure
                truncated (bool): episode is truncated due a success

            Returns:
                float: The reward for the current step
            \"\"\"
            
        """

        self.llm.add_message(prompt)
        self.logger.info(prompt)
        response = self.llm.generate_response(stream=self.isStream)
        response = self.llm.print_Generator_and_return(response)
        response = self._get_code(response)
        reward_func = self._get_runnable_function(response)
        self.reward_functions.append(reward_func)

        return reward_func

    def _get_code(self, response: str) -> str:
        cleaned_response = response.strip("```").replace("python", "").strip()
        if "def " not in cleaned_response:
            raise ValueError(  #TODO traiter cette erreur
                "La réponse ne contient pas de définition de fonction valide."
            )
        self.logger.info("Code nettoyé pour compilation :\n" + cleaned_response)
        return cleaned_response

    def _get_runnable_function(self, response: str, error: str = None) -> Callable:
        if error is not None:
            self.llm.add_message(error)
            response = self.llm.generate_response(stream=self.isStream)
            response = self.llm.print_Generator_and_return(response)
            response = self._get_code(response)
        try:
            reward_func = self._compile_reward_function(response)
            state, _ = self.env.reset()
            action = self.learning_method.output(state)
            next_observation, _ , terminated, truncated, _ = self.env.step(action)
            self._test_reward_function(
                reward_func,
                observations=next_observation,
                terminated=terminated,
                truncated=truncated,
            )
        except SyntaxError as e:
            self.logger.warning(f"Error syntax {e}")
            return self._get_runnable_function(response, str(e))
        except RuntimeError as e:
            self.logger.warning(f"Error execution {e}")
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
        exec_globals["np"] = np
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
            self.logger.debug(f"Reward function output: {reward}")
        except Exception as e:
            raise RuntimeError(f"Error during reward function execution: {e}")

    def self_refine_reward(  # TODO make it work
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
        refined_response = self.llm.generate_response(stream=self.isStream)
        refined_response = self.llm.print_Generator_and_return(refined_response)

        return self._compile_reward_function(refined_response)

    def evaluate_policy(
        self,
        objectives_metrics: List[callable] = [],
        num_episodes: int = 100,
        visual: bool = False,  # TODO utiliser les parametre
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
        raw_policy, raw_perfs, raw_sr, raw_nb_ep = self.learning_method.train(
            save_name=f"model/raw_{self.learning_method}_{self.env.spec.name}.model",
        )
        policy, perfs, sr, nb_ep = self.learning_method.train(
            self.reward_functions[-1],
            save_name=f"model/{self.learning_method}_{self.env.spec.name}{len(self.reward_functions)}.model",
        )
        raw_states, raw_rewards, raw_sr_test = self.test_policy(raw_policy)
        states, rewards, sr_test = self.test_policy(policy, self.reward_functions[-1])
        # TODO penser aux métrics objective propre à l'environnment, une raison de faire une class environnement extend de celle de base ?
        perso_raw_states = []
        perso_states = []
        
        for objective_metric in objectives_metrics:
            perso_raw_states = objective_metric(raw_states)
            perso_states = objective_metric(states)

        for i in range(len(perso_raw_states)):
            self.logger.info(
                f"{perso_raw_states[i][0]} : human {perso_raw_states[i][1]} llm {perso_states[i][1]}"
            )
        
        self.logger.info(
            "the policy with human reward:"
            + f"\n- during the train: SR {raw_sr}, nb_ep {raw_nb_ep}"
            + f"\n- and during the test: SR {raw_sr_test}\n"
        )
        self.logger.info(
            "the policy with llm reward:"
            + f"\n- during the train: SR {sr}, nb_ep {nb_ep}"
            + f"\n- and during the test: SR {sr_test}\n"
        )

    def test_policy(
        self,
        policy,
        reward_func=None,
        nb_episodes=100,
        max_t=1000,
    ) -> list:
        all_rewards = []
        all_states = []
        nb_success = 0
        x_max = 0
        x_min = float("+inf")
        for epi in range(1, nb_episodes + 1):
            total_reward = 0
            state, _ = self.env.reset()
            for i in range(1, max_t + 1):
                action = policy.output(state)
                next_observation, reward, terminated, truncated, _ = self.env.step(
                    action
                )
                if reward_func is not None:
                    reward = reward_func(next_observation, terminated, truncated)
                total_reward += reward
                state = next_observation
                all_states.append(state)
                if terminated:
                    break
                if truncated:
                    nb_success += 1
                    break
            all_rewards.append(total_reward)
            if total_reward > x_max:
                x_max = total_reward
            if total_reward < x_min:
                x_min = total_reward
        all_rewards = [
            x - x_min / x_max - x_min for x in all_rewards
        ]  # Min-Max normalized
        return all_states, all_rewards, (nb_success / nb_episodes)

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
