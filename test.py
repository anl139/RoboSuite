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
# Suppress Robosuite warnings
logging.getLogger("robosuite").setLevel(logging.ERROR)

controller_config = load_composite_controller_config(controller=None, robot="PANDA")

env = robosuite.make(
    "Door",
    robots=["Panda"],
    gripper_types="default",
    controller_configs=controller_config,
    has_renderer=False,
    render_camera="frontview",
    has_offscreen_renderer=False,
    control_freq=20,
    horizon=200,
    use_object_obs=True,
    use_camera_obs=False,
)
KEYS = [
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "handle_pos",
    "door_pos",
    "hinge_qpos",
]

def flatten_obs(obs_dict, keys=KEYS):
    """Return a 1D numpy vector built from obs_dict in a stable order."""
    parts = []
    for k in keys:
        v = obs_dict[k]
        parts.append(np.asarray(v).ravel())
    return np.concatenate(parts).astype(np.float32)

obs0 = env.reset()
sample_vec = flatten_obs(obs0)
input_dims = sample_vec.size  # must match the Agent input_dims you passed
actor_learning_rate = 0.001
critic_learning_rate = 0.001
batch_size = 128
layer_1_size = 256
layer_2_size = 128
obs_spec = env.observation_spec()
agent = Agent(actor_learning_rate=actor_learning_rate,critic_learning_rate=critic_learning_rate,tau=0.005,input_dims=input_dims,
              env=env, n_actions=env.action_dim, layer1_size=layer_1_size, layer2_size=layer_2_size,batch_size=batch_size)
writer = SummaryWriter('logs')
n_games = 10000
best_score = 0
episode_identifier = f"0 - actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} layer_1_size={layer_1_size} layer_2_size={layer_2_size}"

obs = env.reset()  # obs is an OrderedDict

# Flatten the values into a single vector
state_vector = np.concatenate([v.ravel() for v in obs.values()])

print("Observation vector shape:", state_vector.shape)
agent.load_models()

for i in range(n_games):
    obs_dict = env.reset()
    observation = flatten_obs(obs_dict)   # ✅ vector
    done = False
    score = 0.0
  
    while not done:
        action = agent.choose_action(observation)
        action = np.asarray(action, dtype=np.float32)

        next_obs_dict, reward, done, info = env.step(action)
        next_observation = flatten_obs(next_obs_dict)  # ✅ vector

        agent.remember(
            observation,          # ✅ vector
            action,               # ✅ vector
            float(reward),
            next_observation,     # ✅ vector
            float(done),
        )

        agent.learn()
        observation = next_observation
        score += reward

    print(f"Episode {i} | Score {score}")