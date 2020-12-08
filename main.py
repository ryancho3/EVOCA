
from Population import Population
import os
import time
from multiprocessing import Pool
import multiprocessing
import pandas as pd
r = [0.35, 0.15, 0.4, 0.4, 0.3, 0.2, 0.3, 0.3, 0.4, 0.4, 0.3, 0.25]
p = [0.037, 0.015, 0.02, 0.03, 0.03, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.01]
K = 12
T = 242
size = 100
rate = 0.02
version = 0.2
g = 1000
cores = int((multiprocessing.cpu_count())/2)
hyperthreading = True
if hyperthreading:
    cores = int(multiprocessing.cpu_count())

def clear():
    time.sleep(0.3)
    os.system('cls')

def worker(ID):
    r_bestf = []
    r_bestex = []
    r_init = []
    r_avg = []
    r_gain = []
    r_mavg = []
    r_mgain = []
    iteration = 0
    print("Worker " + str(ID) + " initialized, starting tasks.")
    for j in range(int(96/cores)):
        print("Worker " + str(ID) + " starting iteration " + str(iteration))
        pop = Population(K, T, r, p, size, rate)
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
            """print("Generation: " + str(gen))
            print("Average fitness: " + str(average))
            print("\u0394Average Fitness From Previous Gen: " + str(average - prev))
            print("Total Average Fitness Gain/Loss: " + str(gain))"""
            prev = average
            best = pop.getbestagent()
            """print("")
            print("best agent: " + str(best.expression))
            print("best agent fitness: " + str(best.fitness))
            print("best agent total: " + str(total(best.expression)))"""

        r_bestf.append(best.fitness)
        r_bestex.append(best.expression)
        r_init.append(init)
        r_avg.append(average)
        r_gain.append(gain)
        r_mavg.append(maxaverage)
        r_mgain.append(maxgain)
        iteration += 1
    print("Worker " + str(ID) + " completed tasks.")
    return [r_bestf, r_bestex, r_init, r_avg, r_gain, r_mavg, r_mgain]




if __name__ == '__main__':
    t = time.time()
    p = Pool(processes=cores)
    data = p.map(worker, [k for k in range(cores)])
    p.close()
    bestf = []
    bestex = []
    init = []
    avg = []
    gain = []
    mavg = []
    mgain = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            bestf.extend(data[i][0])
            bestex.extend(data[i][1])
            init.extend(data[i][2])
            avg.extend(data[i][3])
            gain.extend(data[i][4])
            mavg.extend(data[i][5])
            mgain.extend(data[i][6])

    percent = int(rate * 100)
    results = {'Best Agent': bestex, 'Best Fitness': bestf, 'Initial P Avg Fitness': init, 'P Avg Fitness': avg,
               'P Gain': gain, 'Max Recorded P Avg Fitness': mavg, 'Max Recorded P Gain': mgain}
    frame = pd.DataFrame(results)
    frame.to_csv('RawData/' + str(percent) + '.csv')
    elapsed = time.time() - t
    print("Process completed in " + str(elapsed))








