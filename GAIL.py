from crowd_sim.envs.crowd_sim_sgan import CrowdSimSgan
from crowd_sim.envs.crowd_sim_sgan_apf import CrowdSimSganApf
from crowd_sim.envs.crowd_sim_no_pred import CrowdSimNoPred
import gym
import time
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C
from imitation_learning import rollout

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm

from arguments import get_args
from crowd_nav.configs.config import Config

def make_env(seed, rank, env_config, envNum=1):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = CrowdSimSganApf()
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.configure(env_config)
        env.seed(seed + rank)
        env.setup(seed=seed+rank, num_of_env=envNum)
        return env

    return _init


def main():
    config = Config()

    num_cpu = 1  # Number of processes to use
    seed = 0

    venv = SubprocVecEnv([make_env(seed, i, config, num_cpu) for i in range(num_cpu)])
    venv = VecTransposeImage(venv)

    rng = np.random.default_rng()

    rollouts = rollout.rollout(
        policy=None,
        venv=venv,
        sample_until=rollout.make_sample_until(min_timesteps=None, min_episodes=100),
        rng=rng,
        unwrap=False,
        exclude_infos=True,
        verbose=True
    )

    learner = PPO(
        env=venv,
        policy='CnnPolicy',
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
    )

    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
    )

    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=256,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True
    )

    gail_trainer.train(300000)

    learner.save('./train/GAIL/model1')

    gail_trainer.train(300000)

    learner.save('./train/GAIL/model2')

if __name__ == '__main__':
    main()
