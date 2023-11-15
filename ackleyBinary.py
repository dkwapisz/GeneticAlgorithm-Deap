import math
import random
from random import randint
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools

a_coeff = 20
b_coeff = 0.2
c_coeff = 2 * math.pi

binary_length = 20
variables = 2
decimal_digits_precision = 5

min_fitness_values = []
max_fitness_values = []
avg_fitness_values = []
std_fitness_values = []

def individual(icsl):
    genome = list()
    for x in range(0, variables * binary_length):
        genome.append(randint(0, 1))
    return icsl(genome)


def decodeInd(individual):
    decoded_numbers = []

    for i in range(0, len(individual), binary_length):
        integer_part = sum([bit * 2 ** j for j, bit in enumerate(individual[i:i + binary_length // 2][::-1])])
        decimal_part = sum([bit * 2 ** j for j, bit in enumerate(individual[i + binary_length // 2:i + binary_length][::-1])]) / 10 ** decimal_digits_precision

        decoded_numbers.append(integer_part + decimal_part)

    return decoded_numbers


def ackleyFitnessFunction(individual):
    ind = decodeInd(individual)
    x = ind[0]
    y = ind[1]
    N = 2

    sum1 = x ** 2 + y ** 2
    sum2 = math.cos(c_coeff * x) + math.cos(c_coeff * y)

    term1 = -a_coeff * math.exp(-b_coeff * math.sqrt(1 / N * sum1))
    term2 = -math.exp(1 / N * sum2)

    result = term1 + term2 + a_coeff + math.e

    return result,


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("individual", individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", ackleyFitnessFunction)
toolbox.register("select", tools.selRoulette)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

sizePopulation = 200
probabilityMutation = 0.3
probabilityCrossover = 0.3
numberIteration = 100

pop = toolbox.population(n=sizePopulation)
fitnesses = list(map(toolbox.evaluate, pop))

for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

numberElitism = 1

g = 0

while g < numberIteration:
    g = g + 1
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    listElitism = []

    for x in range(0, numberElitism):
        listElitism.append(tools.selBest(pop, 1)[0])

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        # cross two individuals with probability CXPB
        if random.random() < probabilityCrossover:
            toolbox.mate(child1, child2)
            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        # mutate an individual with probability MUTPB
        if random.random() < probabilityMutation:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print(" Evaluated %i individuals" % len(invalid_ind))

    pop[:] = offspring + listElitism

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    min_fitness_values.append(min(fits))
    max_fitness_values.append(max(fits))
    avg_fitness_values.append(mean)
    std_fitness_values.append(std)

    print(" Min %s" % min(fits))
    print(" Max %s" % max(fits))
    print(" Avg %s" % mean)
    print(" Std %s" % std)
    best_ind = tools.selBest(pop, 1)[0]

print("Best individual is %s" % (decodeInd(best_ind),))
print("Fitness value: %s" % best_ind.fitness.values)
print("-- End of (successful) evolution --' -")

generations = list(range(1, numberIteration + 1))

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(generations, min_fitness_values, label='Min Fitness', color='red')
plt.xlabel('Generation')
plt.ylabel('Fitness Values')
plt.legend()
plt.title('Minimum Fitness Progression')

plt.subplot(2, 2, 2)
plt.plot(generations, max_fitness_values, label='Max Fitness', color='green')
plt.xlabel('Generation')
plt.ylabel('Fitness Values')
plt.legend()
plt.title('Maximum Fitness Progression')

plt.subplot(2, 2, 3)
plt.plot(generations, avg_fitness_values, label='Avg Fitness', color='blue')
plt.xlabel('Generation')
plt.ylabel('Fitness Values')
plt.legend()
plt.title('Average Fitness Progression')

plt.subplot(2, 2, 4)
plt.plot(generations, std_fitness_values, label='Std Fitness', color='purple')
plt.xlabel('Generation')
plt.ylabel('Fitness Values')
plt.legend()
plt.title('Standard Deviation Fitness Progression')

plt.tight_layout()
plt.show()