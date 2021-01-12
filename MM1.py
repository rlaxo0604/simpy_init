''' M/M/1 '''





import simpy
import numpy as np




def generate_interarrival():
    return np.random.exponential(1./5.0)

def generate_service():
    return np.random.exponential(1./4.0)

def factory_run(env, servers):
    i = 0
    while True:
        i += 1
        yield env.timeout(generate_interarrival())
        print(env.now, 'Lot arrival')
        env.process(lot(env, i, servers))

def lot(env, lot, servers):
    with servers.request() as request:
        print(env.now, 'Lot {} arrives'.format(lot))
        yield request
        print(env.now, 'lot {} is loaded'.format(lot))
        yield env.timeout(generate_service())
        print(env.now, 'lot {} departs'.format(lot))

np.random.seed(0)

env = simpy.Environment()
servers = simpy.Resource(env, capacity=1)

env.process(factory_run(env, servers))
env.run(until=50)