# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 13:09:12 2023

@author: Tafadzwa RJ Mheuka
"""

import tensorflow as tf

# Define input layer
board_input = tf.keras.Input(shape=(19, 19, 17))

# Define convolutional layers
conv1 = tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu')(board_input)
conv2 = tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu')(conv1)
conv3 = tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu')(conv2)

# Define "policy head" layers
policy_conv = tf.keras.layers.Conv2D(2, 1, padding='same', activation='relu')(conv3)
policy_flat = tf.keras.layers.Flatten()(policy_conv)
policy_output = tf.keras.layers.Dense(19*19, activation='softmax')(policy_flat)

# Define "value head" layers
value_conv = tf.keras.layers.Conv2D(1, 1, padding='same', activation='relu')(conv3)
value_flat = tf.keras.layers.Flatten()(value_conv)
value_hidden = tf.keras.layers.Dense(256, activation='relu')(value_flat)
value_output = tf.keras.layers.Dense(1, activation='tanh')(value_hidden)

# Define model
model = tf.keras.Model(inputs=board_input, outputs=[policy_output, value_output])

'''
This code defines a neural network with several convolutional layers, 
followed by two "heads" that predict the probability of each move and the expected outcome of the game, respectively.
 The model is trained using reinforcement learning, 
where it learns to play the game of Go by playing against itself and improving its strategy over time.
'''

import numpy as np

# Define the game environment
num_states = 10
num_actions = 4
rewards = np.random.rand(num_states, num_actions)

# Define the Q-table
q_table = np.zeros((num_states, num_actions))

# Set hyperparameters
num_episodes = 1000
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# Run the Q-learning algorithm
for episode in range(num_episodes):
    state = 0
    done = False
    while not done:
        # Choose an action using epsilon-greedy policy
        if np.random.uniform() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table[state, :])
        # Take the action and observe the next state and reward
        next_state = np.random.randint(num_states)
        reward = rewards[state, action]
        # Update the Q-value for the current state-action pair
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
        # Move to the next state
        state = next_state
        # Check if the game is over
        if state == num_states - 1:
            done = True

# Test the trained Q-table
state = 0
done = False
