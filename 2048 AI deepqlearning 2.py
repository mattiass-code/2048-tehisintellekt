from tkinter import Frame, Label, CENTER

import matplotlib.pyplot as plt
from matplotlib import style

import numpy as np
from keras.models import Sequential, load_model
#import keras.backend.tensorflow_backend as backend
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

current_points = 0
maximum_reward = 0
reward = 0

##############################
##############################

##parameeter##
DISCOUNT = 0.99#0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 50#50  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

EXTRA_INFO = ''
MIN_REWARD_C = 0  # For model save
MEMORY_FRACTION = 0.20#0.20

# Environment settings
EPISODES = 10_000 ##parameeter##
EPISODE_LABEL = "10k"

SAMPLE_SIZE = 20
MODULE_VALUE = EPISODES / SAMPLE_SIZE

# Exploration settings
##parameeter##
epsilon = 1  # not a constant, going to be decayed 
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.005

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

######################################
######################################

def maximum_value(list, hm):
    oned_list = []
    for row in list:
        for element in row:
            oned_list.append(element)

    summa = 0
    for i in range(hm):
        summa += max(oned_list)
        oned_list.remove(max(oned_list))

    return summa

def action_choice(choice):
    if choice == 0:
        return c.KEY_UP
    if choice == 1:
        return c.KEY_DOWN
    if choice == 2:
        return c.KEY_LEFT
    if choice == 3:
        return c.KEY_RIGHT


def new_game(n):
    matrix = []

    for i in range(n):
        matrix.append([0] * n)
    return matrix


def add_two(mat):
    a = random.randint(0, len(mat)-1)
    b = random.randint(0, len(mat)-1)
    while(mat[a][b] != 0):
        a = random.randint(0, len(mat)-1)
        b = random.randint(0, len(mat)-1)
    mat[a][b] = 2
    return mat


def game_state(mat):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 2048:
                return 'win'
    for i in range(len(mat)-1):
        # intentionally reduced to check the row on the right and below
        for j in range(len(mat[0])-1):
            if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                return 'not over'
    for i in range(len(mat)):  # check for any zero entries
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
                return 'not over'
    for k in range(len(mat)-1):  # to check the left/right entries on the last row
        if mat[len(mat)-1][k] == mat[len(mat)-1][k+1]:
            return 'not over'
    for j in range(len(mat)-1):  # check up/down entries on last column
        if mat[j][len(mat)-1] == mat[j+1][len(mat)-1]:
            return 'not over'
    return 'lose'


def reverse(mat):
    new = []
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0])-j-1])
    return new


def transpose(mat):
    new = []
    for i in range(len(mat[0])):
        new.append([])
        for j in range(len(mat)):
            new[i].append(mat[j][i])
    return new


def cover_up(mat):

    new = []
    for j in range(c.GRID_LEN):
        partial_new = []
        for i in range(c.GRID_LEN):
            partial_new.append(0)
        new.append(partial_new)
    done = False
    for i in range(c.GRID_LEN):
        count = 0
        for j in range(c.GRID_LEN):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return (new, done)


def merge(mat):

    done = False
    global current_points 
    current_points = 0
    global merged
    merged = 0

    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN-1):
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                mat[i][j] *= 2
                current_points += mat[i][j]
                merged += 1
                mat[i][j+1] = 0
                done = True

    return (mat, done)


def up(game):
    # return matrix after shifting up
    game = transpose(game)
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = transpose(game)
    return (game, done)


def down(game):
    # return matrix after shifting down
    game = reverse(transpose(game))
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = transpose(reverse(game))
    return (game, done)


def left(game):
    # return matrix after shifting left
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    return (game, done)


def right(game):
    # return matrix after shifting right
    game = reverse(game)
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = reverse(game)
    return (game, done)

def tühjad(mat):

    tühjad = 0

    for i in mat:
        for j in i:
            if j == 0:
                tühjad += 1

    return tühjad



######################################
######################################

class c(): #game constants

    SIZE = 400
    GRID_LEN = 4
    GRID_PADDING = 10

    BACKGROUND_COLOR_GAME = "#b0815f"
    BACKGROUND_COLOR_CELL_EMPTY = "#bb8e6e"

    BACKGROUND_COLOR_DICT = {2: "#f7e7d6", 4: "#f4d0a7", 8: "#ecbd7d",
                             16: "#f1a765", 32: "#fd9258", 64: "#ee7442",
                             128: "#e6ca79", 256: "#ebc363", 512: "#ffb44d",
                             1024: "#ed9b3f", 2048: "#f76046",

                             4096: "#f7e7d6", 8192: "#e9b21d", 16384: "#fb953c",
                             32768: "#f95353", 65536: "#f6768c", }

    CELL_COLOR_DICT = {2: "#a5724e", 4: "#a5724e", 8: "#f9f6f2", 16: "#f9f6f2",
                       32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2",
                       256: "#f9f6f2", 512: "#f9f6f2", 1024: "#f9f6f2",
                       2048: "#f9f6f2",

                       4096: "#a5724e", 8192: "#f9f6f2", 16384: "#f9f6f2",
                       32768: "#f9f6f2", 65536: "#f9f6f2", }

    FONT = ("Comfortaa", 40, "bold")

    KEY_UP_ALT = "\'\\uf700\'"
    KEY_DOWN_ALT = "\'\\uf701\'"
    KEY_LEFT_ALT = "\'\\uf702\'"
    KEY_RIGHT_ALT = "\'\\uf703\'"

    KEY_UP = "'w'"
    KEY_DOWN = "'s'"
    KEY_LEFT = "'a'"
    KEY_RIGHT = "'d'"
    KEY_BACK = "'b'"

    KEY_J = "'j'"
    KEY_K = "'k'"
    KEY_L = "'l'"
    KEY_H = "'h'"


class GameGrid(Frame):

    ##parameeter##
    SIZE = 4
    WIN_REWARD = 1_000_000
    LOSE_PENALTY = -5000
    #MOVE_PENALTY = -5

    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    ACTION_SPACE_SIZE = 4

    d = {0: (110, 142, 187),
     2: (214, 231, 247),
     4: (167, 208, 244),
     8: (125, 189, 236),
     16: (101, 167, 241),
     32: (88, 146, 253),
     64: (66, 116, 238),
     128: (121, 202, 230),
     256: (99, 195, 235),
     512: (77, 180, 255),
     1024: (63, 155, 237),
     2048: (70, 96, 247),}

    def __init__(self):
        Frame.__init__(self)

        self.grid()

        self.master.title('2048')

        self.commands = {c.KEY_UP: up, c.KEY_DOWN: down,
                         c.KEY_LEFT: left, c.KEY_RIGHT: right,
                         c.KEY_UP_ALT: up, c.KEY_DOWN_ALT: down,
                         c.KEY_LEFT_ALT: left, c.KEY_RIGHT_ALT: right,
                         c.KEY_H: left, c.KEY_L: right,
                         c.KEY_K: up, c.KEY_J: down}
        
        self.grid_cells = []


        self.init_matrix()
        self.update_grid_cells()



    def reset(self):
        global maximum_reward
        global reward

        self.episode_step = 0
        self.matrix = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        self.history_matrixs = list()
        self.matrix = add_two(self.matrix)
        new_episode = False
        round_number = 0
        self.update_grid_cells()

        observation = np.array(self.get_image())
        return(observation)

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size

        for y, row in enumerate(self.matrix):
            for x, element in enumerate(row):
                env[x][y] = self.d[element]
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


    def init_grid(self):

        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME,
                           width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(background, bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                             width=c.SIZE / c.GRID_LEN,
                             height=c.SIZE / c.GRID_LEN)
                cell.grid(row=i, column=j, padx=c.GRID_PADDING,
                          pady=c.GRID_PADDING)
                t = Label(master=cell, text="",
                          bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                          justify=CENTER, font=c.FONT, width=5, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)


    def gen(self):
        return random.randint(0, c.GRID_LEN - 1)


    def init_matrix(self):
        self.matrix = new_game(c.GRID_LEN)
        self.history_matrixs = list()
        self.matrix = add_two(self.matrix)
        self.matrix = add_two(self.matrix)


    def update_grid_cells(self):

        if self.history_matrixs == []:
            self.history_matrixs.append(self.matrix) # add initial grid

        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]

        self.update_idletasks()


    def key_down(self, action):

        global new_episode
        global reward
        global maximum_reward
        reward = 0

        move = action_choice(action) # next move

        if move in self.commands: 

            finished = False
            while not finished:

                self.matrix, done = self.commands[move](self.matrix) # move

                if done:
                    finished = True
                    self.matrix = add_two(self.matrix)
                    self.history_matrixs.append(self.matrix) # record last move
                    self.update_grid_cells()

                    done = False

                    if game_state(self.matrix) == 'win':

                        new_episode = True
                        reward = self.WIN_REWARD

                    if game_state(self.matrix) == 'lose':

                        new_episode = True
                        reward = self.LOSE_PENALTY
                else:
                    action = np.random.randint(0, 4)
                    move = action_choice(action)

    def generate_next(self):
        index = (self.gen(), self.gen())

        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2


    def step(self, action):
        self.episode_step += 1
        self.key_down(action)

        new_observation = np.array(self.get_image())

        global reward
        global maximum_reward

        if reward != self.WIN_REWARD and reward != self.LOSE_PENALTY:
            ##parameeter##
            reward = maximum_value(self.matrix, 1)*5 + (maximum_value(self.matrix, 3)-maximum_value(self.matrix, 1))*4 + current_points*2 + tühjad(self.matrix)*50

        done = False
        if reward == self.WIN_REWARD or reward == self.LOSE_PENALTY:
            ##parameeter##
            last_reward = maximum_value(self.matrix, 1)*5 + (maximum_value(self.matrix, 3)-maximum_value(self.matrix, 1))*2 + current_points*0.5 + tühjad(self.matrix)*50
            done = True

        
        return new_observation, reward, done


env = GameGrid()

# For stats
ep_rewards = []

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(f"{EPISODE_LABEL}_{EXTRA_INFO}", int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):


        # model = load_model("models/alt4_300k_150620.48avg_631612.00max_22104.00min_1615332006_.model") #To load previously used model
        model = Sequential()

        ##parameeter##
        model.add(Conv2D(256, (1, 1), input_shape=(4, 4, 3)))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Conv2D(256, (1, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255

        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]


agent = DQNAgent()

# Iterate over episodes
episodenr = 0
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    episodenr += 1

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)

    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])

        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD_C:

            agent.model.save(f'models/{EPISODE_LABEL}_{average_reward:_>7.2f}avg_{max_reward:_>7.2f}max_{min_reward:_>7.2f}min_{int(time.time())}_{EXTRA_INFO}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
