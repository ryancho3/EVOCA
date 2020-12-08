
from Population import Population
import os
import time
import tkinter as tk

def clear():
    time.sleep(0.3)
    os.system('cls')



if __name__ == '__main__':
    version = 0.1
    r = [0.35, 0.15, 0.4, 0.4, 0.3, 0.2, 0.3, 0.3, 0.4, 0.4, 0.3, 0.25]
    p = [0.037, 0.015, 0.02, 0.03, 0.03, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.01]
    K = 12
    T = 242
    window = tk.Tk()
    size = 100
    rate = 0.02
    pop = Population(K, T, r, p, size, rate)
    pop.initialize()

    init = pop.getaveragefitness()
    prev = 0
    label = tk.Label(text="EVOCA v" + str(version))
    label.pack()
    window.mainloop()
    for i in range(1000):
        clear()
        pop.calcallfitness()
        average = pop.getaveragefitness()
        print("average fitness: " + str(average))
        print("\u0394Average Fitness From Previous Gen: " + str(average - prev))
        print("Total Average Fitness Gain/Loss: " + str(average-init))
        prev = average
        best = pop.getbestagent()
        print("")
        print("best agent: " + str(best.expression))
        print("best agent fitness: " + str(best.fitness))
        pop.setpopulation(pop.nextgeneration())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
