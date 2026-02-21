import time
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite
from robosuite.controllers import load_composite_controller_config
import numpy as np
from network import *
from buffer import *
from td3_torch import Agent
import logging
from robosuite.wrappers.gym_wrapper import GymWrapper
# Suppress Robosuite warnings
logging.getLogger("robosuite").setLevel(logging.ERROR)

controller_config = load_composite_controller_config(controller=None, robot="PANDA")

env = robosuite.make(
    "Door",
    robots=["Panda"],
    gripper_types="default",
    controller_configs=controller_config,
    has_renderer=True,
    render_camera="frontview",
    has_offscreen_renderer=False,
    control_freq=20,
    horizon=200,
    use_object_obs=True,
    use_camera_obs=False,
    reward_shaping=True
)
env = GymWrapper(env)
actor_learning_rate = 0.001
critic_learning_rate = 0.001
batch_size = 128
layer_1_size = 256
layer_2_size = 128
agent = Agent(actor_learning_rate=actor_learning_rate,critic_learning_rate=critic_learning_rate,tau=0.005,input_dims=env.observation_space.shape,
              env=env, n_actions=env.action_space.shape[0], layer1_size=layer_1_size, layer2_size=layer_2_size,batch_size=batch_size)
writer = SummaryWriter('logs')
n_games = 3
best_score = 0
episode_identifier = f"0 - actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} layer_1_size={layer_1_size} layer_2_size={layer_2_size}"
print(env.reward_range)
agent.load_models()
for i in range(n_games):
    observation, _ = env.reset()
    observation = np.array(observation, dtype=np.float32)
    done = False
    score = 0.0

    while not done:
        action = agent.choose_action(observation, validation=True)
        action = np.array(action, dtype=np.float32)

        step_result = env.step(action)
        next_observation, reward, terminated, truncated, info = step_result
        done = terminated or truncated

        next_observation = np.array(next_observation, dtype=np.float32)
        env.render()
        score += reward
        observation = next_observation
        time.sleep(0.03)


    writer.add_scalar('Score', score, i)
    if i % 20 == 0:
        agent.save_models()
        print(f"Episode {i} | Score {score}")

writer.close()
