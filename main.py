import random as rd
import matplotlib.pyplot as plt
import numpy as np
import pygame
from typing import Union
from pygame.locals import *

TIME_PER_GENERATION = 4
POPULATION_SIZE = 500

MAX_FOOD = 50

FOOD_HELP = 4
FOOD_HARVEST = 6
FOOD_STEAL = 6
FOOD_MIN = 10


class Simulation:
    def __init__(self):
        pygame.init()
        self.DISPLAYSURF = pygame.display.set_mode((300,300))

        fps = pygame.time.Clock()
        fps.tick(60)


class Blob:
    def __init__(self, genome=(0, 0, 0)):

        self.genome = [a for a in genome]
        self.food = 0
        if genome == (0, 0, 0):
            for i in range(len(self.genome)):
                self.genome[i] = rd.uniform(0, 1)

    def fitness(self):
        return max(self.food - (self.genome[0] * FOOD_MIN), 0)

    def get_food(self):
        if self.genome[1] <= 0.5:
            self.food += self.genome[0] * FOOD_HARVEST

    def steal(self, other: 'Blob'):
        if self.genome[1] > 0.5:
            if other.food >= FOOD_STEAL:
                self.food += FOOD_STEAL * self.genome[1]
                other.food -= FOOD_STEAL
            else:
                self.food += other.food
                other.food = 0

    def help(self, other: 'Blob'):
        if self.genome[2] > 0.5 and other.genome[2] > 0.5:
            if self.food >= self.genome[2] * FOOD_HELP:
                self.food -= self.genome[2] * FOOD_HELP
                other.food += self.genome[2] * FOOD_HELP


def selection_pair(population: list):
    fitness = [blob.fitness() for blob in population]
    return rd.choices(population=population, weights=fitness, k=2)


def crossover(a: Blob, b: Blob):
    p = rd.randint(1, len(a.genome) - 1)

    a_genome = a.genome[0:p] + b.genome[p:]
    b_genome = b.genome[0:p] + a.genome[p:]

    return Blob(a_genome), Blob(b_genome)


def mutation(blob: Blob):
    options = [-0.1, 0.1, 0.05, -0.05]
    genome = blob.genome

    index = rd.randint(0, len(genome)-1)
    change = rd.randint(0, len(options)-1)
    genome[index] += options[change]

    if genome[index] < 0:
        genome[index] = 0
    elif genome[index] > 1:
        genome[index] = 1

    return Blob(genome)


def run_evolution(time_per_gen: int, population_size: int, generation_limit: int, verbose=True):

    # populate first generation
    population = []
    stealers_records = np.array([], dtype=int)
    fittness_avgs = np.array([], dtype=int)
    producers = []
    helpers = []

    for _ in range(population_size):
        population.append(Blob())

    for gen in range(generation_limit):

        for i in range(time_per_gen):
            for blob in population:
                blob.get_food()
                blob.steal(rd.choice(population))
                blob.help(rd.choice(population))


        x_fittness = [i for i in range(len(population))]
        blob_fitness = [blob.fitness() for blob in population]
        fittness_avg = sum(blob_fitness) / len(blob_fitness)
        fittness_avgs = np.append(fittness_avgs, fittness_avg)

        if verbose:
            fig, ax = plt.subplots()  # Create a figure containing a single axes.

            ax.plot(x_fittness, blob_fitness, marker='o')  # Plot some data on the axes.
            ax.plot(x_fittness, [fittness_avg for i in x_fittness], "r--", label="avg")
            ax.set_xlabel("blob index")
            ax.set_ylabel("fitness")
            plt.legend()
            plt.show()

            fig, (ax1, ax2, ax3) = plt.subplots(3,1)  # Create a figure containing a single axes.
            ax1.plot([i for i in range(len(population))], sorted([blob.genome[0] for blob in population]), marker='x', color="red")
            ax1.set_xlabel("blob index")
            ax2.plot([i for i in range(len(population))], sorted([blob.genome[1] if blob.genome[1] > 0.5 else 0 for blob in population]), marker='o')  # Plot some data on the axes.

            ax3.plot([i for i in range(len(population))], sorted([blob.genome[2] for blob in population]), marker='v', color="green")
            ax3.plot([i for i in range(population_size)], [0.5 for i in range(population_size)], "r--")
            plt.show()

        count_stealer = 0
        count_producers = 0
        count_helpers = 0
        for blob in population:
            if blob.genome[1] > 0.5:
                count_stealer += 1
            elif blob.genome[1] < 0.5:
                count_producers += 1
            if blob.genome[2] > 0.5:
                count_helpers += 1

        stealers_records = np.append(stealers_records, count_stealer)
        producers.append(count_producers)
        helpers.append(count_helpers)

        next_generation = []
        try:
            a, b = selection_pair(population)

            a.food = 0
            b.food = 0
            next_generation.append(a)
            next_generation.append(b)

            for j in range(int(len(population) / 2) - 1):
                parent_a, parent_b = selection_pair(population)

                offspring_a, offspring_b = crossover(parent_a, parent_b)
                offspring_a = mutation(offspring_a)
                offspring_b = mutation(offspring_b)

                next_generation += [offspring_a, offspring_b]

            population = next_generation
        except IndexError:
            break

    show_population_over_time(fittness_avgs, 'fittness level')
    show_population_over_time(stealers_records, 'num of stealers')
    show_population_over_time(producers, 'num of producers')
    show_population_over_time(helpers, 'num of helpers')


def show_population_over_time(record: Union[np.numarray, list], ylabel: str, xlabel: str='generations', marker='o'):
    plt.style.use("fivethirtyeight")
    x = [i for i in range(len(record))]

    # calculate the trend line
    z = np.polyfit(x, record, 1)
    p = np.poly1d(z)

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(x, record, marker=marker)
    ax.plot(x, p(x), "r--", lw=2, label="trend line")  # Plot the trend line
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.legend()
    plt.show()


def main():

    run_evolution(TIME_PER_GENERATION, POPULATION_SIZE, 100, False)


if __name__ == '__main__':
    main()
