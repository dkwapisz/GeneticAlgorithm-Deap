import random
from random import randint

from deap import base
from deap import creator
from deap import tools

binary_length = 20
variables = 2
decimal_digits_precision = 5

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


def fitnessFunction(individual):
    ind = decodeInd(individual)
    return ((ind[0] + 2 * ind[1] - 7) ** 2 + (2 * ind[0] + ind[1] - 5) ** 2),


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("individual", individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitnessFunction)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

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

print("Best individual is %s" % (decodeInd(best_ind),))
print("Fitness value: %s" % best_ind.fitness.values)
print("-- End of (successful) evolution --' -")
