# EVOlutionary Capacity Allocation algorithm v0.2
# Ryan Cho
from functools import partial

from Population import Population
import time
from multiprocessing import Pool
import multiprocessing
import pandas as pd

# Global Variables
r = [0.35, 0.15, 0.4, 0.4, 0.3, 0.2, 0.3, 0.3, 0.4, 0.4, 0.3, 0.25]  # Probabilities of Repair
p = [0.037, 0.015, 0.02, 0.03, 0.03, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.01]  # Probablilites of Failure
K = 12  # Number of Machines
T = 275  # Total Buffer Capacity
size = 100  # Population Size
rate = 0.01  # Mutation Rate
version = 0.2
multitest = False
g = 1000  # Generation per Iteration
cores = int((multiprocessing.cpu_count()) / 2)  # Number of Physical CPU Cores
hyperthreading = True
if hyperthreading:
    cores = int(multiprocessing.cpu_count())  # Number of Virtualization CPU Cores


def worker(ID, rt):  # Worker for Multiprocessing Parallelization
    r_bestf = []  # Best Fitness Values
    r_bestex = []  # Expressions of Best Fitness Agents
    r_init = []
    r_avg = []
    r_gain = []
    r_mavg = []
    r_mgain = []
    iteration = 1
    print("Worker " + str(ID) + " initialized, starting tasks.")
    for j in range(int(96 / cores)):
        print("Worker " + str(ID) + " started iteration " + str(iteration) + " for rate " + str(rt) + ".")
        pop = Population(K, T, r, p, size, rt)
        pop.initialize()
        gen = 0
        init = pop.getaveragefitness()
        prev = 0
        maxaverage = 0
        maxgain = 0
        for i in range(g):
            if i != 0:
                pop.setpopulation(pop.nextgeneration())
            gen += 1
            pop.calcallfitness()
            average = pop.getaveragefitness()
            if average > maxaverage:
                maxaverage = average
            gain = average - init
            if gain > maxgain:
                maxgain = gain
            prev = average
            best = pop.getbestagent()

        r_bestf.append(best.fitness)
        r_bestex.append(best.expression)
        r_init.append(init)
        r_avg.append(average)
        r_gain.append(gain)
        r_mavg.append(maxaverage)
        r_mgain.append(maxgain)
        iteration += 1
    print("Worker " + str(ID) + " completed tasks for rate " + str(rt))
    return [r_bestf, r_bestex, r_init, r_avg, r_gain, r_mavg, r_mgain]



if __name__ == '__main__':
    rte = 0.1
    for i in range(1):

        t = time.time()
        p = Pool(processes=cores)
        args1 = [k for k in range(cores)]
        args2 = [rte] * cores
        data = p.starmap(worker, zip(args1, args2))
        p.close()
        bestf = []
        bestex = []
        init = []
        avg = []
        gain = []
        mavg = []
        mgain = []
        for i in range(len(data)):
            bestf.extend(data[i][0])
            bestex.extend(data[i][1])
            init.extend(data[i][2])
            avg.extend(data[i][3])
            gain.extend(data[i][4])
            mavg.extend(data[i][5])
            mgain.extend(data[i][6])



        percent = int(rte * 100)
        results = {'Best Agent': bestex, 'Best Fitness': bestf, 'Initial P Avg Fitness': init, 'P Avg Fitness': avg,
                   'P Gain': gain, 'Max Recorded P Avg Fitness': mavg, 'Max Recorded P Gain': mgain}
        frame = pd.DataFrame(results)
        frame.to_csv('RawData/' + str(T) + "_" + str(percent) + '.csv')
        elapsed = time.time() - t
        print("Process completed in " + str(elapsed))
        rte += 0.01