#problem.py
import random
import math
from setup import Setup
class Problem:
    def __init__(self):
        self._solution = None
        self._value = 0
        self._numEval = 0
        self.pFileName = ''
        self.bestSolution = None
        self.bestMinimum = None
        self.avgMinimum = None
        self.avgNumEval = None
        self.sumOfNumEval = None
        self.avgWhen = None
        self.pType = None

    def setVariables(self, parameters):
        pass

    def randomInit(self):
        pass

    def evaluate(self):
        pass

    def mutants(self):
        pass

    def randomMutant(self, current):
        pass

    def describe(self):
        pass

    def storeResult(self, solution, value):
        self._solution = solution
        self._value = value

    def getSolution(self):
        return self._solution

    def getValue(self):
        return self._value

    def getNumEval(self):
        return self._numEval

    def storeExpResult(self):
        pass

    def report(self):
        print()
        print("Solution found:")
        print(self.getSolution())
        print("Minimum value: {0:,.3f}".format(self.getValue()))
        print("Total number of evaluations: {0:,}".format(self.getNumEval()))

    def setProblemType(self, pType):
        self.pType = pType


class Numeric(Problem):
    def __init__(self):
        super().__init__()
        self._expression = ''
        self._domain = []     # domain as a list

    def setVariables(self, parameters):
        ## Read in a TSP (# of cities, locatioins) from a file
        ## Then, set the relevant class variables
        fileName = parameters['pFileName']
        infile = open(fileName, 'r')
        # First line is number of cities
        self._numCities = int(infile.readline())
        cityLocs = []
        line = infile.readline()  # The rest of the lines are locations
        while line != '':
            cityLocs.append(eval(line)) # Make a tuple and append
            line = infile.readline()
        infile.close()
        self._locations = cityLocs
        self._distanceTable = self.calcDistanceTable()


    def randomInit(self): # Return a random initial point as a list
        domain = self._domain
        low, up = domain[1], domain[2]
        init = []
        for i in range(len(low)):              # For each variable
            r = random.uniform(low[i], up[i])  # take a random value
            init.append(r)
        return init  # list of values

    def evaluate(self, current):
        ## Evaluate the expression of 'p' after assigning
        ## the values of 'current' to the variables
        self._numEval += 1
        varNames = self._domain[0]
        for i in range(len(varNames)):
            assignment = varNames[i] + '=' + str(current[i])
            exec(assignment)
        return eval(self._expression)

    def mutants(self, current):
        neighbors = []
        for i in range(len(current)):  # For each variable
            mutant = self.mutate(current, i, self.delta)
            neighbors.append(mutant)
            mutant = self.mutate(current, i, -self.delta)
            neighbors.append(mutant)
        return neighbors

    def mutate(self, current, i, d): ## Mutate i-th of 'current' if legal
        mutant = current[:]   # Make a copy of 'current'
        domain = self._domain # [VarNames, low, up]
        l = domain[1][i]      # Lower bound of i-th
        u = domain[2][i]      # Upper bound of i-th
        if l <= (mutant[i] + d) <= u:
            mutant[i] += d
        return mutant

    def randomMutant(self, current):
        # Pick a random locus
        i = random.randint(0, len(current) - 1)
        # Mutate the chosen locus
        if random.uniform(0, 1) > 0.5:
            d = self.delta
        else:
            d = -self.delta
        return self.mutate(current, i, d)

    def takeStep(self, x, v): # v=f(x) Take gradient and make update if legal
        grad = self.gradient(x, v)  # Gradient at point 'x'
        xCopy = x[:]
        for i in range(len(xCopy)):
            xCopy[i] = xCopy[i] - self.alpha * grad[i]
        if self.isLegal(xCopy):  # Check if 'xCopy' is within the domain
            return xCopy
        else:
            return x

    def gradient(self, x, v): # 'x' is a vector (list of valules)
        grad = []   # Calculate partial derivatives and combine them
        for i in range(len(x)):
            xCopyH = x[:]
            xCopyH[i] += self.dx
            g = (self.evaluate(xCopyH) - v) / self.dx
            grad.append(g)
        return grad

    def isLegal(self, x):   # Check if 'x' is within the domain
        domain = self._domain      # [VarNames, low, up]
        low = domain[1]   # Lower bounds
        up = domain[2]    # Upper bounds
        flag = True
        for i in range(len(low)):
            if x[i] < low[i] or up[i] < x[i]:
                flag = False
                break
        return flag

    def describe(self):
        print()
        print("Objective function:")
        print(self._expression)
        print("Search space:")
        varNames = self._domain[0] # domain: [VarNames, low, up]
        low = self._domain[1]
        up = self._domain[2]
        for i in range(len(varNames)):
            print("Variable: {0}; Range: [{1}, {2}]".format(varNames[i], low[i], up[i]))

    def report(self):
        print()
        print("Best order of visits:")
        for i in range(0, len(self._solution), 10):  # 10개씩 출력
            print("   " + " ".join(str(x) for x in self._solution[i:i+10]))
        print("Minimum tour cost: {0:,}".format(int(self._value)))
        super().report()

class Tsp(Problem):
    def __init__(self):
        super().__init__()
        self._numCities = 0
        self._locations = []       # A list of tuples
        self._distanceTable = []

    def setVariables(self, parameters):
        ## Read in a TSP (# of cities, locatioins) from a file
        ## Then, set the relevant class variables
        fileName = parameters['pFileName']
        infile = open(fileName, 'r')
        # First line is number of cities
        self._numCities = int(infile.readline())
        cityLocs = []
        line = infile.readline()  # The rest of the lines are locations
        while line != '':
            cityLocs.append(eval(line)) # Make a tuple and append
            line = infile.readline()
        infile.close()
        self._locations = cityLocs
        self._distanceTable = self.calcDistanceTable()


    def calcDistanceTable(self):
        locations = self._locations
        table = []
        for i in range(self._numCities):
            row = []
            for j in range(self._numCities):
                dx = locations[i][0] - locations[j][0]
                dy = locations[i][1] - locations[j][1]
                d = round(math.sqrt(dx**2 + dy**2), 1)
                row.append(d)
            table.append(row)
        return table # A symmetric matrix of pairwise distances

    def randomInit(self):   # Return a random initial tour
        n = self._numCities
        init = list(range(n))
        random.shuffle(init)
        return init

    def evaluate(self, current):
        ## Calculate the tour cost of 'current'
        ## 'current' is a list of city ids
        self._numEval += 1
        n = self._numCities
        table = self._distanceTable
        cost = 0
        for i in range(n - 1):
            locFrom = current[i]
            locTo = current[i+1]
            cost += table[locFrom][locTo]
        #처음으로 돌아가기
        cost += table[locTo][0]
        return cost

    def mutants(self, current): # Inversion only
        n = self._numCities
        neighbors = []
        count = 0
        triedPairs = []
        while count <= n:  # Pick two random loci for inversion
            i, j = sorted([random.randrange(n) for _ in range(2)])
            if i < j and [i, j] not in triedPairs:
                triedPairs.append([i, j])
                mutant = self.inversion(current, i, j)
                count += 1
                neighbors.append(mutant)
        return neighbors

    def inversion(self, current, i, j):  ## Perform inversion
        mutant = current[:]  # Make a copy of 'current'
        while i < j:
            mutant[i], mutant[j] = mutant[j], mutant[i]
            i += 1
            j -= 1
        return mutant

    def randomMutant(self, current): # Inversion only
        while True:
            i, j = sorted([random.randrange(self._numCities) for _ in range(2)])
            if i < j:
                mutant = self.inversion(current, i, j)
                break
        return mutant

    def describe(self):
        print()
        print("Number of cities:", self._numCities)
        print("City locations:")
        print(self._locations)

    def storeExpResult(self, result, optimalValue):
        self.bestSolution = result
        self.bestMinimum = optimalValue

    def report(self):
        print()
        print("Best order of visits:")
        for i in range(0, len(self.bestSolution), 10):  # 10개씩 출력
            print("   " + " ".join(str(x) for x in self.bestSolution[i:i+10]))
        print("Minimum tour cost: {0:,}".format(int(self.bestMinimum)))
        super().report()
