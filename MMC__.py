import numpy as np
import simpy
from model.baseinfo import BaseInfo
from model.lot import Lot
from model.arrival import Arrival

'''
M/M/c
dispatching rule: FIFO
'''

MEAN_ARRIVAL = 2
MEAN_SERVICE = 8

NUM_MACHINES = 3

base_info = BaseInfo()
arrivals = Arrival(base_info)

class Machine:
    def __init__(self, name):
        self.name = name

'''
Arrival 제너레이터
매 exp(MEAN_ARRIVAL)마다 LOT을 생성(env.yield 함수)
생성시 저장되는 이름: lot i
생성후 이를 arrival_buffer에 저장(store.put 함수)
'''

def lot_arrival(env, arrival_buffer):
    while True:
        lot_arrival_time = np.random.exponential(MEAN_ARRIVAL)

        yield env.timeout(lot_arrival_time)
        i = arrivals.rand_pick()
        lot = Lot(i, base_info)
        yield arrival_buffer.put(lot)
        print('{:.2f}'.format(env.now), ' : ' + lot.product_type + ' arrived')
        print('arrival buffer:', arrival_buffer.items)

'''
machine 제너레이터
여러개의 machine에 대하여 각각의 service가 독립적으로 돌아가도록 생성
env.timeout(0)은 machines를 generator로 만들기 위함(없으면 안돌아감)
'''

def machines(env, arrival_buffer, n):
    for i in range(0,n):
        yield env.timeout(0)
        machine = lot_service(env, "machine{:2d}".format(i), arrival_buffer)
        env.process(machine)

'''
각각의 생성된 machine에 대하여
서비스를 예약하는 함수
매 exp(MEAN_SERVICE)
'''

def lot_service(env, machine_name, arrival_buffer):
    while True:
        yield env.timeout(0)
        lot = yield arrival_buffer.get()
        print('{:.2f}'.format(env.now), ' : ' + machine_name + ' start ' + lot.product_type)

        service_time = np.random.exponential(MEAN_SERVICE)
        yield env.timeout(service_time)
        print('{:.2f}'.format(env.now), ' : ' + machine_name + ' end ' + lot.product_type)

env = simpy.Environment()

arrival_buffer = simpy.Store(env)

arrival = env.process(lot_arrival(env,arrival_buffer))

env.process(machines(env,arrival_buffer,NUM_MACHINES))

env.run(until=100)