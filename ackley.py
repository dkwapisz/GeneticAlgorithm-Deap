import math
import random

from deap import base
from deap import creator
from deap import tools

a_coeff = 20
b_coeff = 0.2
c_coeff = 2 * math.pi


def individual(icsl):
    genome = list()
    for x in range(0, 2):
        genome.append(random.uniform(-32, 32))
    return icsl(genome)


def ackleyFitnessFunction(individual):
    x, y = individual
    N = 2

    sum1 = x ** 2 + y ** 2
    sum2 = math.cos(x * c_coeff) + math.cos(y * c_coeff)

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
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.5, indpb=0.5)

sizePopulation = 100
probabilityMutation = 0.2
probabilityCrossover = 0.8
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

    print(" Min %s" % min(fits))
    print(" Max %s" % max(fits))
    print(" Avg %s" % mean)
    print(" Std %s" % std)
    best_ind = tools.selBest(pop, 1)[0]

print("Best individual is %s" % (best_ind,))
print("Fitness value: %s" % best_ind.fitness.values)
print("-- End of (successful) evolution --' -")
