import numpy as np
import simpy

MEAN_ARRIVAL = 2
MEAN_SERVICE = 8.5

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
def default_dispathcing(machines):
    queue_length = []
    # machine에 들어온 request를 카운트
    for i in range(NUM_MACHINES):
         queue_length.append(machines[i].count)
    # request = 0 인 machine: 현재 가동중이지 않음
    # request = 1 인 machine: 현재 가동중
    shortest = np.argmin(queue_length)
    # 가동중이지 않은 machine을 선택
    machine = machines[shortest]
    return machine, shortest

def lot_arrival(env, arrival_buffer):
    i = 1
    while True:
        #lot_arrival_time = np.random.exponential(MEAN_ARRIVAL)
        lot_arrival_time = (MEAN_ARRIVAL)
        lot = 'lot {} '.format(i)
        yield env.timeout(lot_arrival_time)
        yield arrival_buffer.put(lot)

        #print('Arrival buffer: ', arrival_buffer.items)
        print(' {} arrived: '.format(lot), env.now)
        i += 1

def factory_run(env, arrival_buffer, machines):
    yield env.timeout(0)
    machine, machine_num = default_dispathcing(machines)
    a = lot_service(env, machine, machine_num, arrival_buffer)
    env.process(a)

def lot_service(env, machine, machine_num, arrival_buffer):
    while True:
        #service_time = np.random.exponential(MEAN_SERVICE)
        service_time = (MEAN_SERVICE)

        if machine.count < 1:
            with machine.request() as req:
                yield req
                lot = yield arrival_buffer.get()
                print('{} is loaded at {} machine: '.format(lot, machine_num), env.now)
                #print('Arrival buffer after loaded: ', arrival_buffer.items)
                yield env.timeout(service_time)
                print('{} serviced at {} machine: '.format(lot, machine_num), env.now)
        else:
            print('!! all machines are occupied !!')


env = simpy.Environment()
arrival_buffer = simpy.Store(env)
machines = [simpy.Resource(env, capacity=1) for _ in range(NUM_MACHINES)]

arrival = env.process(lot_arrival(env, arrival_buffer))

run = env.process(factory_run(env, arrival_buffer, machines))

env.run(until=30)
'''
machine, machine_num = default_dispathcing(machines)
if machine.count < 1:
    env.process(lot_service(env, machine, machine_num))
else:
    print('!! all machines are occupied !!')
'''