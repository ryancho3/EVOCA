import numpy as np
import random
import copy as copy


class Agent:

    def __init__(self, k, t, r, p):
        self.K = k
        self.T = t
        self.r = r
        self.p = p
        self.expression = [0] * k
        self.fitness = 0.0

    def initialize(self):
        for i in range(self.K):
            self.expression[i] = random.randint(0, int((self.T / self.K) + 10))

        self.adjust()

    def setexpression(self, expression):
        self.expression = copy.deepcopy(expression)

    def adjust(self):

        tot = self.total()

        while tot > self.T:
            rindex = random.randint(0, len(self.expression) - 1)
            while (self.expression[rindex] == 0):
                rindex = random.randint(0, len(self.expression) - 1)

            if (self.expression[rindex] != 0):
                self.expression[rindex] = self.expression[rindex] - 1
            tot = self.total()

        while tot < self.T:
            rindex = random.randint(0, len(self.expression) - 1)
            self.expression[rindex] = self.expression[rindex] + 1
            tot = self.total()

    def total(self):
        total = 0
        for i in range(len(self.expression)):
            total += self.expression[i]
        return total

    def mutate(self):
        total = 0
        index1 = random.randint(0, len(self.expression) - 1)
        index2 = random.randint(0, len(self.expression) - 1)
        if index1 == index2:
            return
        total = self.expression[index1] + self.expression[index2]
        mutation1 = random.randint(0, total)
        mutation2 = total - mutation1
        self.expression[index1] = mutation1
        self.expression[index2] = mutation2

    def TwoMOneB(self, N, r1, r2, p1, p2):
        # Initialization
        Y1 = (r1 + r2 - r1 * r2 - r1 * p2) / (p1 + p2 - p1 * p2 - p1 * r2)
        Y2 = (r1 + r2 - r1 * r2 - p1 * r2) / (p1 + p2 - p1 * p2 - r1 * p2)
        X = Y2 / Y1;
        C = 1

        # 2 probability of boundary states calculation
        p001 = C * X * (r1 + r2 - r1 * r2 - r1 * p2) / (r1 * p2)
        p100 = C * X
        p101 = C * X * Y2
        p111 = C * X * (r1 + r2 - r1 * r2 - r1 * p2) / (p2 * (p1 + p2 - p1 * p2 - r1 * p2))
        pN100 = C * X ** (N - 1)
        pN110 = C * (X ** (N - 1)) * Y1
        pN111 = C * (X ** (N - 1)) * (r1 + r2 - r1 * r2 - p1 * r2) / (p1 * (p1 + p2 - p1 * p2 - p1 * r2))
        pN10 = C * (X ** (N - 1)) * (r1 + r2 - r1 * r2 - p1 * r2) / (p1 * r2)
        pBound = p001 + p100 + p101 + p111 + pN100 + pN110 + pN111 + pN10

        # 3 probability of N = n or internal states calculation
        d = 1e-3
        pInt = 0

        if X != 1:
            pInt = ((C * ((X ** (N - 1)) - X ** 2) / (X - 1)) * ((1 + Y1) * (1 + Y2)))
            a = -2 * X + (N - 1) * (X ** (N - 2))
            b = (X ** (N - 1)) - X ** 2
            c = X - 1
        else:
            for i in range(2, N - 2):
                pInt = (0.5 * C * N * (N - 3) * (1 + Y1) * (1 + Y2))

        # 4 normalization
        C = 1 / (pBound + pInt)
        # New C

        # 5 probability of boundary states calculation after normalization
        p001 = C * X * (r1 + r2 - r1 * r2 - r1 * p2) / (r1 * p2)
        p100 = C * X
        p101 = C * X * Y2
        p111 = C * X * (r1 + r2 - r1 * r2 - r1 * p2) / (p2 * (p1 + p2 - p1 * p2 - r1 * p2))
        pN100 = C * X ** (N - 1)
        pN110 = C * (X ** (N - 1)) * Y1
        pN111 = C * (X ** (N - 1)) * (r1 + r2 - r1 * r2 - p1 * r2) / (p1 * (p1 + p2 - p1 * p2 - p1 * r2))
        pN10 = C * (X ** (N - 1)) * (r1 + r2 - r1 * r2 - p1 * r2) / (p1 * r2)

        # 6 probability of boundary states after normalization
        #pBound = p001 + p100 + p101 + p111 + pN100 + pN110 + pN111 + pN10
        ps = p001
        pb = pN10

        d = 1e-3

        if X != 1:
            #pInt = ((C * ((X ** (N - 1)) - X ** 2) / (X - 1)) * ((1 + Y1) * (1 + Y2)))
            a = (-2 * X) + (N - 1) * (X ** (N - 2))
            b = (X ** (N - 1)) - X ** 2
            c = X - 1
            nInt = (C * (X * (a - (b / c))) / c) * (1 + Y1) * (1 + Y2)
        else:
            #pInt = (0.5 * C * N * (N - 3) * (1 + Y1) * (1 + Y2))
            nInt = 0.5 * C * N * (N - 3) * (1 + Y1) * (1 + Y2)

        # 8 performance calculation
        e = r1 / (r1 + p1)
        P = e * (1 - pN10)
        n_bar = p100 + p101 + p111 + nInt + (N - 1) * (pN100 + pN110 + pN111) + N * pN10

        return [P, n_bar, ps, pb]

    def calcfitness(self):
        # Decomposition Method for long line machine
        # DDX algorithm
        # K = number of machine
        # N = number of buffer per each slot
        # r = probability of repair
        # p = probability of failure
        multiplier = 100

        pd = [[None] * (self.K - 1)] * multiplier
        rd = [[None] * (self.K - 1)] * multiplier
        pu = [[None] * (self.K - 1)] * multiplier
        ru = [[None] * (self.K - 1)] * multiplier
        e = [[None] * (self.K)] * multiplier
        E = [[None] * (self.K - 1)] * multiplier
        n = [[None] * (self.K - 1)] * multiplier
        ps = [[None] * (self.K - 1)] * multiplier
        pb = [[None] * (self.K - 1)] * multiplier
        Iu = [[None] * (self.K - 1)] * multiplier
        X = [[None] * (self.K - 1)] * multiplier
        Id = [[None] * (self.K - 1)] * multiplier
        Y = [[None] * (self.K - 1)] * multiplier

        for i in range(0, self.K - 1):
            pd[0][i] = self.p[i + 1]
            rd[0][i] = self.r[i + 1]

        pu[0][0] = self.p[0]
        ru[0][0] = self.r[0]

        for i in range(0, self.K):
            e[0][i] = (self.r[i] / (self.r[i] + self.p[i]))

        count = 0

        for i in range(1, self.K - 1):
            tempret = self.TwoMOneB(self.expression[i - 1], ru[count][i - 1], rd[count][i - 1], pu[count][i - 1],
                                    pd[count][i - 1])
            E[count][i - 1] = tempret[0]
            n[count][i - 1] = tempret[1]
            ps[count][i - 1] = tempret[2]
            pb[count][i - 1] = tempret[3]
            Iu[count][i] = (1 / (E[count][i - 1])) + (1 / e[count][i]) - (pd[count][i - 1] / rd[count][i - 1]) - 2
            X[count][i] = ps[count][i - 1] / (Iu[count][i] * E[count][i - 1])
            ru[count][i] = (ru[count][i - 1] * X[count][i]) + (self.r[i] * (1 - X[count][i]))
            pu[count][i] = Iu[count][i] * ru[count][i]

        for i in range(self.K - 3, -1, -1):
            tempret = self.TwoMOneB(self.expression[i + 1], ru[count][i + 1], rd[count][i + 1], pu[count][i + 1],
                                    pd[count][i + 1])
            E[count][i + 1] = tempret[0]
            n[count][i + 1] = tempret[1]
            ps[count][i + 1] = tempret[2]
            pb[count][i + 1] = tempret[3]
            Id[count][i] = (1 / (E[count][i + 1]) + (1 / e[count][i + 1]) - Iu[count][i + 1] - 2)
            Y[count][i + 1] = pb[count][i + 1] / (Id[count][i] / E[count][i + 1])
            rd[count + 1][i] = (rd[count][i + 1] * Y[count][i + 1]) + (self.r[i + 1] * (1 - Y[count][i + 1]))
            pd[count + 1][i] = Id[count][i] * rd[count][i]

        while (abs(E[count][0] - E[count][1]) >= 0.00001):
            count = count + 1

            pd[count][self.K - 2] = self.p[self.K - 1]
            rd[count][self.K - 2] = self.r[self.K - 1]
            pu[count][0] = self.p[0]
            ru[count][0] = self.r[0]

            for i in range(1, self.K - 1):
                tempret = self.TwoMOneB(self.expression[i - 1], ru[count][i - 1], rd[count][i - 1], pu[count][i - 1],
                                        pd[count][i - 1])
                E[count][i - 1] = tempret[0]
                n[count][i - 1] = tempret[1]
                ps[count][i - 1] = tempret[2]
                pb[count][i - 1] = tempret[3]
                Iu[count][i] = (1 / (E[count][i - 1])) + (1 / e[count][i]) - (pd[count][i - 1] / rd[count][i - 1]) - 2
                X[count][i] = ps[count][i - 1] / (Iu[count][i] * E[count][i - 1])
                ru[count][i] = (ru[count][i - 1] * X[count][i]) + (self.r[i] * (1 - X[count][i]))
                pu[count][i] = Iu[count][i] * ru[count][i]

            for i in range(self.K - 3, -1, -1):
                tempret = self.TwoMOneB(self.expression[i + 1], ru[count][i + 1], rd[count][i + 1], pu[count][i + 1],
                                        pd[count][i + 1])
                E[count][i + 1] = tempret[0]
                n[count][i + 1] = tempret[1]
                ps[count][i + 1] = tempret[2]
                pb[count][i + 1] = tempret[3]
                Id[count][i] = (1 / (E[count][i + 1]) + (1 / e[count][i + 1]) - Iu[count][i + 1] - 2)
                Y[count][i + 1] = pb[count][i + 1] / (Id[count][i] / E[count][i + 1])
                rd[count + 1][i] = (rd[count][i] * Y[count][i + 1]) + (self.r[i + 1] * (1 - Y[count][i + 1]))
                pd[count + 1][i] = Id[count][i] * rd[count][i]

        Eff = E[count][0]
        nbar = n[count]
        pb_ret = pb[count]
        ps_ret = ps[count]
        self.fitness = Eff
        ##return [Eff, nbar, pb_ret, ps_ret]
        return Eff

