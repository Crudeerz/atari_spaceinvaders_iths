import keras
import gymnasium as gym
import ale_py
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing

gym.register_envs(ale_py)


model_file = "../Local/Models/space_qmodel_4535.keras"
agent = keras.models.load_model(model_file)

env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="rgb_array")
env = AtariPreprocessing(env)
env = FrameStack(env, 4)

# prefix the video with episode from modelfile
prefix = model_file.split("_")[2]
prefix += "_video"
env = gym.wrappers.RecordVideo(env, video_folder="./Videos", disable_logger=True, name_prefix=prefix)

state, _ = env.reset()
done = False
while not done:
    # first convert to a tensor for compute efficiency
    state_tensor = keras.ops.convert_to_tensor(state)
    # shape of state is 4, 84, 84, but we need 84, 84, 4
    state_tensor = keras.ops.transpose(state_tensor, [1, 2, 0])
    # Add batch dimension
    state_tensor = keras.ops.expand_dims(state_tensor, 0)
    # ’predict ’ method is for large batches , call as function instead
    action_probs = agent(state_tensor, training=False)
    # Take ’best ’ action
    action = keras.ops.argmax(action_probs[0]).numpy()

    state, reward, done, _, _ = env.step(action)

    
