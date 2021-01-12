import numpy as np
import simpy

MEAN_ARRIVAL = 2
MEAN_SERVICE = 5

NUM_MACHINES = 3
'''Default dispatching rule
1. LOT selection
머신에서 생산이 종료되면, 대기 buffer의 LOT 중 가장 오래된 랏을 선택
== BUFFER의 제일 첫번째 랏
2. Machine selection
다른 머신에는 1의 queue length / 끝난 머신은 0의 queue length
제일 짧은 줄을 찾아서 distpatching
초기에 다 비어있는 경우는 1번 머신부터 채움
'''

def lot_arrival(env, arrival_buffer):
    i = 1
    while True:
        lot_arrival_time = np.random.exponential(MEAN_ARRIVAL)
        #lot_arrival_time = MEAN_ARRIVAL

        yield env.timeout(lot_arrival_time)
        lot_name = "lot{:2d} ".format(i)
        yield arrival_buffer.put(lot_name)
        print(env.now, ' : ' + lot_name + 'arrived')
        print('lots: {}'.format(arrival_buffer.items))
        i += 1

def machines(env, arrival_buffer, n):
    for i in range(0,n):
        yield env.timeout(0)
        machine = lot_service(env, "machine{:2d}".format(i), arrival_buffer)
        env.process(machine)

def lot_service(env, machine_name, arrival_buffer):
    while True:
        yield env.timeout(0)
        waiting_time = env.now
        lot = yield arrival_buffer.get()
        print(env.now, ' : ' + machine_name + ' start ' + lot)

        service_time = np.random.exponential(MEAN_SERVICE)
        #service_time = MEAN_SERVICE
        yield env.timeout(service_time)
        print(env.now, ' : ' + machine_name + ' end ' + lot)

env = simpy.Environment()

arrival_buffer = simpy.Store(env)

arrival = env.process(lot_arrival(env,arrival_buffer))

env.process(machines(env,arrival_buffer,NUM_MACHINES))

env.run(until=100)