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
        #self.gettop500()
        self.calcallfitness()


    def calcallfitness(self):
        for i in range(len(self.agents)):
            self.agents[i].calcfitness()

    def adjustall(self):
        for i in range(len(self.agents)):
            self.agents[i].adjust()
    def setpopulation(self, pop):
        self.agents = pop
        #self.gettop500()

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
        ret = None
        """if (top == True):
            agents = self.top500
            tot = 0
            for i in range(50):
                tot += agents[i].fitness
            threshold = random.random() * tot
            add = 0.0
            for j in range(len(agents)):
                add += agents[j].fitness
                if (add > threshold):
                    ret = agents[i]
        else:"""
        tot = self.calctotalfitness()
        threshold = random.random() * tot

        add = 0.0
        for k in range(self.s):
            add = add + self.agents[k].fitness
            if (add >= threshold):
                return self.agents[k]


    def gettop500(self):
        agentlist = copy.deepcopy(self.agents)
        top500 = []
        for i in range(500):
            max = 0
            maxindex = 0
            for j in range(len(agentlist)):
                if (agentlist[j].fitness > max):
                    max = agentlist[j].fitness
                    maxindex = j
            top500.append(agentlist.pop(maxindex))
        self.top500 = top500
        return top500


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


