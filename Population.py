from Agent import Agent
import copy as copy
import random

class Population:

    def __init__(self, k, t, r, p, size, mutationrate):
        self.K = k
        self.T = t
        self.r = r
        self.p = p
        self.s = size
        self.m = mutationrate
        self.top500 = []

    def initialize(self):
        self.agents = []
        for i in range(self.s):
            self.agents.append(Agent(self.K, self.T, self.r, self.p))
            self.agents[i].initialize()
        self.calcallfitness()


    def calcallfitness(self):
        for i in range(len(self.agents)):
            self.agents[i].calcfitness()

    def adjustall(self):
        for i in range(len(self.agents)):
            self.agents[i].adjust()

    def setpopulation(self, pop):
        self.agents = pop

    def calctotalfitness(self):
        tot = 0.0
        for i in range(len(self.agents)):
            tot = tot + self.agents[i].fitness
        return tot

    def getbestagent(self):

        max = 0
        index = 0
        for i in range (len(self.agents)):
            if (self.agents[i].fitness > max):
                index = i
                max = self.agents[i].fitness
        return self.agents[index]


    def getaveragefitness(self):
        return self.calctotalfitness()/len(self.agents)

    def selection(self):
        tot = self.calctotalfitness()
        threshold = random.random() * tot

        add = 0.0
        for k in range(self.s):
            add = add + self.agents[k].fitness
            if (add >= threshold):
                return self.agents[k]



    def generateoffspring(self):
        parent = self.selection()
        offspring = Agent(self.K, self.T, self.r, self.p)
        offspring.setexpression(parent.expression)
        return offspring



    def nextgeneration(self):
        pop = []
        pop.append(copy.deepcopy(self.getbestagent()))
        for i in range(1, self.s):
            a = self.generateoffspring()
            if (random.random() < self.m):
                a.mutate()
            pop.append(a)

        return pop


