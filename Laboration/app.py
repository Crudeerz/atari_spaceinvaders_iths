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
st.title("Spaceinvaders Laboration AI-23")
st.write("This app is presenting a laboration made for Deep-Learning Course at ITHS. \
          The assignment was to from a given example, use a Deep-Q-learning, reinforcement learning network to train an agent to play the game Atari- Spaceinvaders")
st.write("The app will take you through the necessary code-changes that were made to get the script up and running \
          and some visuals and results of the agent playing the game")
st.divider()

# Model overview
st.write("Here's a quick view of the model that were used for training. The Deep-Q-Network (DQN) is implemented based on the Google Nature Paper from 2015.")
st.markdown("[The paper can be read here](https://arxiv.org/pdf/1312.5602)")
st.image("image.png", caption="Deep Q model (DQN) - used for training")
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
    st.write("**Before change**: shape=(32,4,84,84)")
    st.code("""
                state_sample = np.array([state_history[i] for i in indicies])
            state_next_sample = np.array([state_next_history[i] for i in indicies])
""")
with col2:
    st.write("**After change**: shape=(32,84,84,4)")
    st.code("""
                state_sample = np.moveaxis(np.array([state_history[i] for i in indicies]), 1, -1)
            state_next_sample = np.moveaxis(np.array([state_next_history[i] for i in indicies]), 1, -1)
""")



# Small code changes, list -> deque
st.write("After these changes and other smaller changes as adding functionality for timetracking and saving the modelfile the program runs without errors and exceptions. \
         So some small optimization changes were made.")
st.write("Instead of having to iterate through and appending history information to lists Deques were used. \
         Deques lets you append information to the end of the deque without having to iterate over all indicies before adding data \
         which improves program-speed")




    


st.title("Resultat och grafer")
arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.pyplot(fig)