# optimizer.py

from setup import Setup
from problem import Problem
import random
import math


class Optimizer:
    def __init__(self, problem, numExp):
        self.problem = problem
        self.numExp = numExp
        self.pType = ''
        self.aType = 0
    def setVariables(self, parameters):
        pass

    def getNumExp(self):
        return self.numExp
    
    def getAType(self):
        return self.aType
    
    def displaySetting(self):
        print()
        print("Search algorithm: ", self.__class__.__name__)
        print()
        print("Mutation step size:", self.delta)

class HillClimbing(Optimizer):
    def __init__(self, problem, numExp, pType, limitStuck, numRestart):
        super().__init__(problem, numExp)
        self.limitStuck = limitStuck
        self.numRestart = numRestart

    def run(self):
        pass

    def displaySetting(self):
        super().displaySetting()
        print("Max evaluations with no improvement: ", self.limitStuck)
        print("Number of restarts: ", self.numRestart)

    def randomRestart(self):
        best = self.problem.randomInit()
        bestValue = self.problem.evaluate(best)
        for _ in range(self.numRestart):
            successor = self.run()
            value = self.problem.evaluate(successor)
            if value < bestValue:
                best, bestValue = successor, value
        return best, bestValue

class Stochastic(Optimizer):
    def __init__(self, problem, numExp, limitStuck, numRestart):
        super().__init__(problem, numExp)
        self.limitStuck = limitStuck
        self.numRestart = numRestart
        self.numExp = numExp
        
    def run(self):
        current = self.problem.randomInit()  # 무작위 초기 상태
        valueC = self.problem.evaluate(current)  # 초기 상태의 값
        i = 0
        while i < self.limitStuck:
            neighbors = self.problem.mutants(current)
            successor, valueS = self.stochasticBest(neighbors, self.problem)
            if valueS < valueC:
                current, valueC = successor, valueS
                i = 0  # 개선이 있었으므로 카운터를 리셋
            else:
                i += 1
        return current
    
    def stochasticBest(self, neighbors, p):
        # Smaller values are better in the following list
        valuesForMin = [p.evaluate(indiv) for indiv in neighbors]
        largeValue = max(valuesForMin) + 1
        valuesForMax = [largeValue - val for val in valuesForMin]
        # Now, larger values are better
        total = sum(valuesForMax)
        randValue = random.uniform(0, total)
        s = valuesForMax[0]
        for i in range(len(valuesForMax)):
            if randValue <= s: # The one with index i is chosen
                break
            else:
                s += valuesForMax[i+1]
        return neighbors[i], valuesForMin[i]

    def displaySetting(self):
        super().displaySetting()
        print("Max evaluations with no improvement: ", self.limitStuck)
        print("Number of restarts: ", self.numRestart)
        
class StochasticHillClimbing(HillClimbing):
    def __init__(self, problem, pType, limitStuck, numRestart):
        super().__init__(problem, pType, limitStuck, numRestart)

    def run(self):
        current = self.problem.randomInit()  # 무작위 초기 상태
        valueC = self.problem.evaluate(current)  # 초기 상태의 값
        i = 0
        while i < self.limitStuck:
            neighbors = self.problem.mutants(current)
            successor, valueS = self.stochasticBest(neighbors, self.problem)
            if valueS < valueC:
                current, valueC = successor, valueS
                i = 0  # 개선이 있었으므로 카운터를 리셋
            else:
                i += 1
        return current

    def stochasticBest(self, neighbors, p):
        # Smaller valuse are better in the following list
        valuesForMin = [p.evaluate(indiv) for indiv in neighbors]
        largeValue = max(valuesForMin) + 1
        valuesForMax = [largeValue - val for val in valuesForMin]
        # Now, larger values are better
        total = sum(valuesForMax)
        randValue = random.uniform(0, total)
        s = valuesForMax[0]
        for i in range(len(valuesForMax)):
            if randValue <= s: # The one with index i is chosen
                break
            else:
                s += valuesForMax[i+1]
        return neighbors[i], valuesForMin[i]

class GradientDescent(Optimizer):
    def __init__(self, problem, numExp, limitStuck, numRestart):
        super().__init__(problem, numExp)
        self.limitStuck = limitStuck
        self.numRestart = numRestart

    def run(self):
        current = self.problem.randomInit()  # 무작위 초기 상태
        valueC = self.problem.evaluate(current)  # 초기 상태의 값
        while True:
            successor = self.problem.takeStep(current, valueC)  # 그래디언트 단계
            valueS = self.problem.evaluate(successor)  # 후속 상태의 값
            if valueS >= valueC: break
            else:
                current, valueC = successor, valueS
        self.problem.storeResult(current, valueC)  # 결과 저장

    def displaySetting(self):
        super().displaySetting()
        print("Max evaluations with no improvement: ", self.limitStuck)
        print("Number of restarts: ", self.numRestart)

class FirstChoice(Optimizer):
    def __init__(self, problem, numExp, limitStuck, numRestart):
        super().__init__(problem, numExp)
        self.limitStuck = limitStuck
        self.numRestart = numRestart

    def run(self):
        current = self.problem.randomInit()  # 무작위 초기 상태
        valueC = self.problem.evaluate(current)  # 초기 상태의 값
        i = 0
        while i < self.limitStuck:
            successor = self.problem.randomMutant(current)  # 무작위 이웃
            valueS = self.problem.evaluate(successor)  # 이웃의 값
            if valueS < valueC:
                current, valueC = successor, valueS
                i = 0  # 개선이 있었으므로 카운터를 리셋
            else:
                i += 1
        self.problem.storeResult(current, valueC)  # 결과 저장

    def displaySetting(self):
        super().displaySetting()
        print("Max evaluations with no improvement: ", self.limitStuck)
        print("Number of restarts: ", self.numRestart)

class SteepestAscent(Optimizer):
    def __init__(self, problem, numExp, limitStuck, numRestart):
        super().__init__(problem, numExp)
        self.limitStuck = limitStuck
        self.numRestart = numRestart

    def run(self):
        current = self.problem.randomInit()  # 무작위 초기 상태
        valueC = self.problem.evaluate(current)  # 초기 상태의 값
        while True:
            neighbors = self.problem.mutants(current)  # 이웃 상태들
            if not neighbors: break
            successor = max(neighbors, key=self.problem.evaluate)  # 가장 좋은 이웃
            valueS = self.problem.evaluate(successor)
            if valueS >= valueC: break
            else:
                current, valueC = successor, valueS
        self.problem.storeResult(current, valueC)  # 결과 저장

    def displaySetting(self):
        super().displaySetting()
        print("Max evaluations with no improvement: ", self.limitStuck)
        print("Number of restarts: ", self.numRestart)

class SimulatedAnnealing(Optimizer):
    def __init__(self, problem, numExp=1, limitEval=50000, whenBestFound=None, numSample=None):
        super().__init__(problem, numExp)
        self._limitEval = limitEval
        self._whenBestFound = whenBestFound if whenBestFound is not None else 0
        self._numSample = numSample if numSample is not None else 100
        self.aType = 5  # Or any other integer representing SimulatedAnnealing


    def run(self, problem):
        current = self.problem.randomInit()  # 무작위 초기 상태
        valueC = self.problem.evaluate(current)  # 초기 상태의 값
        best, bestValue = current, valueC
        t = self.initTemp(self.problem)  # 초기 온도 설정
        i = 0
        while i < self._limitEval and t > 0:
            successor = self.problem.randomMutant(current)  # 무작위 이웃
            valueS = self.problem.evaluate(successor)  # 이웃의 값
            if valueS < bestValue:  # 더 좋은 해를 찾았다면
                best, bestValue = successor, valueS
                current, valueC = successor, valueS
                i = 0  # 개선이 있었으므로 카운터를 리셋
            else:
                dE = valueS - valueC
                if random.uniform(0, 1) < math.exp(-dE / t):  # 확률적으로 수락
                    current, valueC = successor, valueS
                i += 1
            t = self.tSchedule(t)  # 온도 감소
        self.problem.storeResult(best, bestValue)  # 결과 저장
        return best

    def initTemp(self, p): # To set initial acceptance probability to 0.5
        diffs = []
        for i in range(self._numSample):
            c0 = p.randomInit()     # A random point
            v0 = p.evaluate(c0)     # Its value
            c1 = p.randomMutant(c0) # A mutant
            v1 = p.evaluate(c1)     # Its value
            diffs.append(abs(v1 - v0))
        dE = sum(diffs) / self._numSample  # Average value difference
        t = dE / math.log(2)        # exp(–dE/t) = 0.5
        return t
    
    def tSchedule(self, t):
        return t * (1 - (1 / 10**4))

    def displaySetting(self):
        print("Search algorithm: ", self.__class__.__name__)
        print("Max evaluations until termination: ", self._limitEval)

        
    def getWhenBestFound(self):
        return self._whenBestFound
    
    def displayNumExp(self):
        print(self.numExp)
