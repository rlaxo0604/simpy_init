import simpy
import numpy as np
from functools import partial, wraps


# dqn
import constants as cst
import random
from collections import deque

from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam



class Photo:
    def __init__(self, env, m, tact, agent, arrivals, states, mask_shop, epsilon):
        self.id = m
        self.env = env
        self.agent = agent
        self.arrivals = arrivals
        self.states = states
        self.mask_name = None
        self.machine = simpy.Store(env, capacity=1)
        self.env.process(self.load_unload())
        print('Now at %.1f,' % env.now, 'M%d generated and started operation.' % m)
        self.process_time = tact
        self.selected_lot = None
        self.mask_shop = mask_shop

        self.epsilon = epsilon

    def select_a_lot(self, action):
        """ Select a lot in the arrival buffer using the action of agent """
        a_lot = 'Lot' + str(action)  # sample code

        # 딕셔너리 형식으로 lot 이름 -> state 상 랏의 idx으로 구현하면 좋을 듯
        return a_lot

    def move_mask(self, current_loc, dest_loc):
        print('Now at %.1f,' % self.env.now, self.mask_name, 'started to move from', current_loc, 'to',
              dest_loc)
        yield self.env.timeout(2.5)  # sample move_time
        self.mask_shop.location[self.mask_name] = dest_loc
        print('Now at %.1f,' % self.env.now, self.mask_name, 'changed location from', current_loc, 'to',
              dest_loc)

    def load_unload(self):
        while True:
            event_load = simpy.Event(self.env)

            if self.selected_lot is not None:

                # Mask resource request & release
                self.mask_name = self.mask_shop.mask_map['M' + str(self.id)]

                with self.mask_shop.masks[self.mask_name].request() as event_print:
                    print('Now at %.1f,' % self.env.now, 'M%d' % self.id, 'requested', self.mask_name)
                    yield event_print
                    print('Now at %.1f,' % self.env.now, 'M%d' % self.id, 'owned', self.mask_name)

                    current_loc = self.mask_shop.location[self.mask_name]
                    dest_loc = self.mask_shop.machine_library['M' + str(self.id)]

                    # Mask move process
                    if current_loc != dest_loc:
                        event_move = self.env.process(self.move_mask(current_loc, dest_loc))
                        yield event_move

                    # Lot loading & unloading process
                    self.states.update_load(self.selected_lot)
                    self.states.update_mask('load') # mask no 추가 필요 / 박사님과 마스크 생각이 다를수 있으니 수정부탁드립니다.

                    # selected_lot -> steptarget idx 변환 작업 필요 or selected lot의 step atrribute 저장 필요
                    self.states.update_step_target(self.selected_lot)

                    yield event_load.succeed(value=(self.id, 'LOAD')) & self.machine.put(self.selected_lot)

                    print('Now at %.1f,' % self.env.now, 'M%d loaded a lot :' % self.id, self.selected_lot)

                    # Lot unloading & training_process
                    event_unload = self.env.timeout(self.process_time, value=(self.id, 'UNLOAD'))
                    yield event_unload
                    unloading_lot = yield self.machine.get()  # get() event의 items로 assign
                    print('Now at %.1f,' % self.env.now, 'M%d unloaded a lot :' % self.id, unloading_lot)
                    self.states.update_mask('unload')

                    # Agent training
                    print('Now at %.1f,' % self.env.now, 'M%d start training :' % self.id)
                    action, next_state = self.agent.train(self.env, self.id, self.states, self.mask_shop, epsilon=1)

                    reward = self.states.get_reward(action)

                # Mask release
                print('Now at %.1f,' % self.env.now, 'M%d' % self.id, 'released', self.mask_name)
                self.states.update_mask('release')



class MaskShop:
    def __init__(self, env, state):
        self.env = env
        self.names = ['R1', 'R2', 'R3']
        self.masks = {mask: simpy.Resource(env, capacity=1) for mask in self.names}
        self.state = state
        self.location = {'R1': 'L1', 'R2': 'L2', 'R3': 'L3'}
        self.machine_library = {'M1': 'L1', 'M2': 'L2', 'M3': 'L3'}
        # self.mask_map = {'M1': 'R1', 'M2': 'R2', 'M3': 'R3'}  # Sample code : 설비별 mask fix case
        self.mask_map = {'M1': 'R1', 'M2': 'R1', 'M3': 'R1'}  # Sample code : 설비별 mask fix case


class DQN:
    def __init__(self, env, id):
        self.env = env
        self.id = id

        self.replay_memory_count = 0
        self.memory = deque(maxlen=1000)
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(cst.F1, input_dim=cst.state_size))
        model.add(LeakyReLU(alpha=cst.ALPHA))
        model.add(Dense(cst.F2))
        model.add(LeakyReLU(alpha=cst.ALPHA))
        model.add(Dense(cst.F3))
        model.add(LeakyReLU(alpha=cst.ALPHA))
        model.add(Dense(cst.F4))
        model.add(LeakyReLU(alpha=cst.ALPHA))
        model.add(Dense(cst.F5))
        model.add(LeakyReLU(alpha=cst.ALPHA))
        model.add(Dense(cst.F6))
        model.add(LeakyReLU(alpha=cst.ALPHA))
        model.add(Dense(cst.F7))
        model.add(LeakyReLU(alpha=cst.ALPHA))
        model.add(Dense(cst.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=cst.LEARN_RATE))
        return model

    def act(self, id, state, epsilon, f_action):
        if np.random.rand() <= epsilon:
            return random.randrange(cst.action_size)
        act_values = self.model.predict(np.reshape(state, (1, cst.state_size)))
        act_values_feasible = act_values * f_action
        for i in range(len(act_values_feasible[0])):
            if act_values_feasible[0][i] == 0:
                act_values_feasible[0][i] -= 100

        return np.argmax(act_values_feasible[0])  # returns action        print('Now at %.1f,' % self.env.now, 'A%d decided an action' % id, action, 'to M' + str(id))

    def feasible_action(self, states, masks):
        f_action = 0

        return f_action

    def train(self, env, id, states, masks, epsilon):
        """ Implement DQN train procedure here including remember() and replay().
            Calculate action based on feasible_action.
            Get next_state and reward from states.
        """
        print('Now at %.1f,' % self.env.now, 'M%d triggers A%d to learn after unload !!!' % (id, id))
        print('States: Get from States and Photo class')

        f_action = self.feasible_action(states, masks)
        current_state = states.current_state()
        action = self.act(id, current_state, epsilon, f_action)
        next_state = states.next_state(current_state, action)

        self.replay_memory_count = min(self.replay_memory_count + 1,
                                                       cst.REPLY_MEMORY_SIZE)
        if self.replay_memory_count == cst.REPLY_MEMORY_SIZE:
            mini_batch = random.sample(self.memory, cst.MINI_BATCH_SIZE)
            state = []
            action = []
            reward = []
            next_state = []
            for i in range(cst.MINI_BATCH_SIZE):
                state.append(mini_batch[i][0])
                action.append(mini_batch[i][1])
                reward.append(mini_batch[i][2])
                next_state.append(mini_batch[i][3])

            target = self.model.predict(np.reshape(state, (cst.MINI_BATCH_SIZE, cst.state_size)))
            t = self.target_model.predict(np.reshape(next_state, (cst.MINI_BATCH_SIZE, cst.state_size)))
            for i in range(len(action)):
                target[i][action[i]] = reward[i] + cst.DISCOUNT * np.amax(t[i])
            self.model.fit(np.reshape(np.array(state), (cst.MINI_BATCH_SIZE, cst.state_size)), target, batch_size=cst.MINI_BATCH_SIZE,
                           epochs=1, verbose=0)

        return action, next_state



class Arrivals:
    def __init__(self, env, state):
        self.env = env
        self.env.process(self.arrive())
        self.state = state

    def arrive(self):
        while True:
            new_lot = np.random.randint(100, 999)
            event_arrive = self.env.timeout(3, value=(new_lot, 'ARRIVAL'))
            yield event_arrive
            self.state.update_arrive(new_lot)
            print('Now at %.1f,' % self.env.now, 'New lot arrived. :', event_arrive.value[0])


class States:
    def __init__(self, env):
        self.env = env

        self.arrival_state = [0] * 1000  # new lot size
        self.step_state = [0] * 1000
        self.mask_state = [0] * len(cst.names)
        self.last_mask_state = [0] * len(cst.names)


    def current_state(self):  # 현재 상태에선 machine state 필요 x, 뻄(PM, failure 추가 시 추가)
        current_state = self.arrival_state + self.step_state + self.mask_state + self.last_mask_state
        return current_state

    def next_state(self, current_state, action):
        '''next state 고려해야할 state는 wip $ step target state'''
        # convert action to wip idx & step idx ( action과 관련된 함수 필요 )
        wip_idx = 0
        step_idx = 0

        if self.arrival_state[wip_idx] > 0:
            temp_arrival_state = self.arrival_state.copy()
            temp_arrival_state[wip_idx] -= 1

            temp_step_state = self.step_state.copy()
            temp_step_state[len(self.arrival_state) + step_idx] -= 1

            next_state = temp_arrival_state + temp_step_state + self.mask_state + self.last_mask_state

            return next_state

        else:
            return current_state

    def get_reward(self, action):
        reward = 0
        return reward

    def update_arrive(self, lot_no):
        pass

    def update_load(self, lot_no):
        pass

    def update_step_target(self, step_no):
        pass

    def update_mask(self, event, mask_no):
        if event == 'load':
            pass
        elif event == 'unload':
            pass
        elif event == 'release':
            pass


class Trace:
    """This class traces an event_name during env.step() and add a schedule_train()
    if the event happens
    """
    def __init__(self, env, event_name):
        self.env = env
        self.machine = None
        self.event_name = event_name

    def trace_step(self, callback, schedule_train):
        """ Replace the 'step()' method of *env* with an event tracing function
        that calls *callbacks* with an events time, priority, ID and
        its instance just BEFORE it is processed.
        """
        def get_env_step_wrapper(env_step_orig, callback, schedule_train):
            """ Generate the wrapper for env.step() """
            @wraps(env_step_orig)
            def trace_step_with_callback():
                """ Call *callback* for the next event if one exist
                BEFORE calling 'env.step()'
                """
                if len(self.env._queue):
                    t, prio, eid, event = self.env._queue[0]
                    # Add callback for trace without changing the original env.step()
                    event_happened = callback(t, prio, eid, event)
                    if event_happened:
                        self.schedule_train(t)
                return env_step_orig(), event_happened
            return trace_step_with_callback

        self.env.step = get_env_step_wrapper(self.env.step, callback, schedule_train)

    def trace_event(self, data, t, prio, eid, event):
        """A callback function, which traces a catch_event"""
        is_unload_event = False
        if (event.value is not None) and (event.value[1] == self.event_name):
            # data.append(t, eid, event.value)  # for trace cum
            is_unload_event = True
            print('Now at %.1f,' % self.env.now, 'Machine', event.value, "traced by callback at %.1f" % t)
            self.machine = event.value[0]
        return is_unload_event

    def schedule_train(self, t):
        """ Write code here if train procedure is implemented outside Photo class"""
        print('Now at %.1f,' % self.env.now, 'A%s' % str(self.machine), 'will learn after unload at %.1f' % t)


def main():
    env0 = simpy.Environment()

    states = States(env0)
    arrivals = Arrivals(env0, states)
    mask_shop = MaskShop(env0, states)


    machine_list = [1, 2, 3]  # sample code
    tact = [3.1, 3.2, 3.3]  # sample code

    agents = {'A' + str(i): DQN(env0, i) for i in machine_list}
    machines = {'M' + str(i): Photo(env0, i, tact[i-1], agents['A' + str(i)], arrivals, states, mask_shop)
                for i in machine_list}

    # Trace Class 활용하는 경우 추가함
    trace = Trace(env0, 'UNLOAD')
    event = None
    trace_event = partial(trace.trace_event, event)
    trace.trace_step(trace_event, trace.schedule_train)

    env0.run(until=20)


if __name__ == main():
    main()
