import streamlit as st 
import matplotlib.pyplot as plt
import numpy as np


magicEnabled = False

# Page settings
st.set_page_config(
    page_title="Atari-SpaceInvaders",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# Title and intro
st.title("Spaceinvaders Laboration :space_invader:")
st.write("This app is presenting a laboration made for Deep-Learning Course at ITHS. \
          The assignment was to from a given example, use a Deep-Q-learning, reinforcement learning network to train an agent to play the game Atari- Spaceinvaders")
st.write("The app will take you through the necessary code-changes that were made to get the script up and running \
          and some visuals and results of the agent playing the game")
st.divider()

# Model overview
st.write("Here's a quick view of the model that were used for training. The Deep-Q-Network (DQN) is implemented based on the Google Nature Paper from 2015.")
st.markdown("[The paper can be read here](https://arxiv.org/pdf/1312.5602)")
st.image("Resources/model.png", caption="Deep Q model (DQN) - used for training")
st.divider()

# Code change section
st.subheader("Code changes")
st.write("When running the program for the first time an exception is thrown:")
st.exception(ValueError("Input 0 of layer 'conv2d' is incompatible with the layer: expected axis -1 of input shape to have value 4, but received input with shape (32, 4, 84, 84)"))
st.write("To get the code working we hade to make sure the shape of the data in the replay buffer matches what the input shape is expecting. To solve this, np.moveaxis was added \
         to rearrange the shape to meet the expected format.")

# Columns for code change comparison
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Before change**: shape=:blue[(32,:red[4],84,84)]")
    st.code("""
                state_sample = np.array([state_history[i] for i in indicies])
            state_next_sample = np.array([state_next_history[i] for i in indicies])
""")
with col2:
    st.markdown("**After change**: shape=:blue[(32,84,84,:green[4])]")
    st.code("""
                state_sample = np.moveaxis(np.array([state_history[i] for i in indicies]), 1, -1)
            state_next_sample = np.moveaxis(np.array([state_next_history[i] for i in indicies]), 1, -1)
""")



# Small code changes, list -> deque
st.text("After these changes and other smaller changes as adding functionality for timetracking and backup-saving of the modelfile, the program runs without errors and exceptions. \
         Furthermore, some small optimization-changes were made.\n \
         Instead of having to iterate through and appending history information to lists Deques were used. \
         Deques lets you append information to the end of the deque without having to iterate over all indicies before adding data \
         which improves program-speed. After changing lists to Deque(), a for loop was implemented to make it look more neat")


col1, col2 = st.columns(2)

with col1:
    st.markdown("**Before change**:")
    st.code(""" xxxxxxx_history = []
    ...""")
    st.code("""
                action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
""")
with col2:
    st.markdown("**After change**:")
    st.code(""" xxxxxxx_history = Deque()
    ...
                    """)
    st.code("""
                history_logs = [action_history, state_history, state_next_history, rewards_history, done_history]
        history_entries = [action, state, state_next, reward, done]
      
        for entry, log in zip(history_entries, history_logs):
            log.append(entry)
""")

st.divider() 

st.header("Follow-up and visuals")
st.write("I had a hard time getting animated pyplots to update simoultaniously as the script was training the model. A test was made to start the matplotlib FuncAnimation \
         in a sepearte Thread as shown in the code example below but appareantley the Main Thread is default for GUI, at this point I decided to move on and take a different approach... :smile:")
st.write("*Example from the code showcasing threading when trying to animate plot:*")
st.code("""
            import threading
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
        
            def update():
                # Function for updating plot
        
            def run_animation():
                 animation = FuncAnimation(fig, update, interval=500)
                 plt.show()

            # Start animation in seperate Thread
            animation_thread = threading.Thread(target=run_animation)
            animation_thread.start()
""")
st.text("Instead of FuncAnimation, I saved updated plots every n:th episode so I could manually construct a .gif of all saved plots and proceed with other tasks")

st.image("Resources/plot.gif", caption="")
st.divider()

# Gameplay section #

st.header("Gameplay")
st.text("Here is a comparison gameplay between agents playing the game at two different trainingstages")
# show agent playing the game
col1, col2 = st.columns(2)

with col1:
    st.text("Gameplay from an early training stage: \n Episode 500")
    st.video("Resources/500.keras_video-episode-0.mp4", autoplay=True, loop=True, muted=True) # Extra *arg (muted) only to generate seperate internal ID for st.video
with col2:
    st.text("Gameplay at a later training stage: \n Episode 4535, more than 24h of training")
    st.video("Resources/4535.keras_video-episode-0.mp4",autoplay=True, loop=True )

# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)

# st.pyplot(fig)