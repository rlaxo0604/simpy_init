"""
Constants for RLS Project
(RLS : Reinforcement Learning Scheduler)
"""
# 나중에 엑셀에서 불러올 cst 값들
names = ['R1', 'R2', 'R3']
n_lot = 1000
n_step = 300
state_size = n_lot + n_step + len(names) * 2
action_size = n_lot


# Experiment Setup
MODE = 'RUN'  #'TRAIN' or 'RUN'
MODE_CODE = {'TRAIN': 'T', 'RUN': 'R'}

EXP_NAME = 'CLUSTERING_mask_update'
EXP_DESC = 'revise_mask_update'
LOG_FILE_NAME = 'RLS_'
LOG_DIRECTORY = 'logs/'

# Input Data Information
DATA_SOURCE = 'data/inputs/xlsx'
#DATA_SOURCE = 'data/inputs/txt_2019_text_mask_prod'
TEMP_DATA_SOURCE = 'Data_orig/'
DATA_TYPE = 'xlsx'

# Machine Information
EVENT = {'ARRIVAL': 0, 'LOAD': 1, 'UNLOAD': 2, 'WAIT': 3, 'CHECK': 4,
         'PM': 5, 'FAIL': 6, 'UP': 7}

MACHINE_LIST = [1, 2, 3]
#MACHINE_LIST = [1, 2, 3 ]
#MACHINE_LIST_TEST = [1, 4]

PM_DURATION = 214.01 * 60
PM_PROB = 0 #5.7 / 90
FAILURE_DURATION = 99 * 60
FAILURE_PROB = 0 #8.8 / 90
FAILURE_DURATION_LIMIT_MIN = 40
WAIT_SECOND = 180

# Arrival Information
INTER_ARRIVAL_SEC = 2 * 60 * 60 / 96.5424  # sec
#INTER_ARRIVAL_SEC_TEST = 2 * 60 * 60 / 10.7269
N_ARRIVAL_LOT = 200
N_ARRIVAL_LOT_TEST = 60

# Simulation parameters
SCHED_HORIZON = 28800

# Model parameters
EXPLORATION_INIT = {'TRAIN': 1, 'RUN': 0}
N_EPISODES = 100  # 10000

REPLY_MEMORY_SIZE = 1000
FINAL_EPSILON = 0
TARGET_UPDATE_FREQUENCY = 100
MINI_BATCH_SIZE = 32
# NO USE # GAMMA = 0.99 # discount rate

F1 = 256
F2 = 512
F3 = 1024
F4 = 512
F5 = 1024
F6 = 512
F7 = 256

ALPHA = 0.1
DISCOUNT = 0.95  # discount rate
LEARN_RATE = 0.000001

NETWORK_PARAM_DIR = 'data/outputs/networks/'

# Plot information
PLOT_FREQUENCY = 500  # 500
GANTT_FREQUENCY = 1  # 500
GANTT_MARKING = 180  # Gantt Change / No_Change marking
GANTT_PATH = 'data/outputs/gantts/'
GRAPH_PATH = 'data/outputs/perfs/'
#GANTT_M_LIST = [1, 2, 3]
GANTT_M_LIST = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Schedule information
SCHED_OUT_DIR = 'data/outputs/scheds/'





