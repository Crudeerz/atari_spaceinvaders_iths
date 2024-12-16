import keras
from keras import layers
import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import numpy as np
import tensorflow as tf
import ale_py
from collections import deque
import datetime
import pathlib


gym.register_envs(ale_py)


render_mode = "rgb_array"
env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode=render_mode)
env = AtariPreprocessing(env)
env = FrameStack(env, 4)

num_actions = env.action_space.n
print(num_actions)

save_model_dir = pathlib.Path("../atari_spaceinvaders_iths/Local/Models/")

if render_mode == "rgb_array":
    trigger = lambda t: t % 1000 == 0
    env = gym.wrappers.RecordVideo(env, video_folder="./Videos", episode_trigger=trigger, disable_logger=True)

def create_q_model():
    return keras.Sequential(
    [
        keras.Input(shape=(84,84,4)),
        layers.Conv2D(32, kernel_size=8, strides=4, activation="relu"),
        layers.Conv2D(64, kernel_size=4, strides=2, activation="relu"),
        layers.Conv2D(64, kernel_size=3, strides=1, activation="relu"),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(num_actions, activation="linear")       
        
    ]
    )


model = create_q_model()
model_target = create_q_model()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Max episodes to run, set to 0 means runt 'til solved
max_episodes = 0
# Max frames to run 
max_frames = 1e7
# Frames to take random actions and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1e6
# Max length of replay buffer
max_memory_length = 2e5
# Abort if more than the below frames are spent in a single game (results in truncated = True)
max_steps_per_episode = 10000
# How often should the action-network be updated
update_after_actions = 4
# How often should the Q-network be cloned from our action network?
update_target_network = 10000
# Use Huber loss for stability (specifically for Adam)
loss_function = keras.losses.Huber()


maxlen = int(max_memory_length)
action_history = deque()
state_history = deque()
state_next_history = deque()
rewards_history = deque()
episode_reward_history = deque(maxlen=100)
running_reward = []
done_history = deque()
episode_count = 0
train_count = 0
frame_count = 0



gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min
batch_size = 32

start_time = datetime.datetime.now()
print(f"Starting training at: {start_time}")

while True:
    observation, _ = env.reset()
    state = np.array(observation)
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = keras.ops.convert_to_tensor(state)
            state_tensor = keras.ops.transpose(state_tensor, [1,2,0])
            state_tensor = keras.ops.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = keras.ops.argmax(action_probs[0]).numpy()
        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _, _ = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        history_logs = [action_history, state_history, state_next_history, rewards_history, done_history]
        history_entries = [action, state, state_next, reward, done]
      
        for entry, log in zip(history_entries, history_logs):
            log.append(entry)

        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            # Get indices of samples for replay buffers
            indicies = np.random.choice(
                range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.moveaxis(np.array([state_history[i] for i in indicies]), 1, -1)
            state_next_sample = np.moveaxis(np.array([state_next_history[i] for i in indicies]), 1, -1)
            rewards_sample = [rewards_history[i] for i in indicies]
            action_sample = [action_history[i] for i in indicies]
            done_sample = keras.ops.convert_to_tensor([float(done_history[i]) for i in indicies])

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample, verbose=0)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma*keras.ops.amax(future_rewards, axis=1)

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample - done_sample)

            # Create a mask so we only calculate loss on the updated Q-values
            masks = keras.ops.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = keras.ops.sum(
                    keras.ops.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            model_target.set_weights(model.get_weights())
            print(f"best score of last 100: {np.max(episode_reward_history)}, running_reward(mean last 100): {running_reward} at episode {episode_count}, frame {frame_count}")
        # Saving model every 300th episode    
        if episode_count % 300 == 0:
            model.save(f"{save_model_dir}/space_qmodel_{episode_count}.keras") #C:\Users\rasmu\Rasmus\VS Code Project\ITHS\atari_spaceinvaders_iths\Local\Models 
        # Print details and time info
        if frame_count % 10000 == 0:
            print(f"{frame_count} frames done at {datetime.datetime.now()}, UP-TIME: {datetime.datetime.now() - start_time}")
            

        # Limit the state and reward history
        if len(rewards_history)>max_memory_length:
            rewards_history.popleft()
            state_history.popleft()
            state_next_history.popleft()
            action_history.popleft()
            done_history.popleft()

        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    # Consider solved if runnig reward exceeds 500
    if running_reward > 500:
        print(f"Solved at episode {episode_count}!")
        model.save(f"space_qmodel_solved.keras")
        break
    if (max_episodes > 0 and episode_count >= max_episodes):
         print(f"Stopped at episode {episode_count}!")
         break
    if (max_frames > 0 and frame_count>=max_frames):
        print(f"Stopped at frame {frame_count}!")
        break
