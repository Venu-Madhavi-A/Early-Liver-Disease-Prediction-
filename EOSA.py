import random
import time
import numpy as np


# EOSA function
def EOSA(population_size, fitness_func,  lb, ub, num_generations):
    N, num_candidates = population_size.shape[0], population_size.shape[1]
    # Create initial population
    population = [[random.uniform(0, 1) for j in range(num_candidates)] for i in range(population_size)]

    Convergence_curve = np.zeros((num_generations, 1))

    t = 0
    ct = time.time()
    for gen in range(num_generations):
        # Evaluate fitness for each individual in population
        fitness_scores = [fitness_func(individual) for individual in population]

        # Select candidates for each election
        candidates = []
        for i in range(num_candidates):
            # Choose candidates at random, with probability proportional to their fitness score
            candidate_scores = [fitness_scores[j] for j in range(population_size) if j not in candidates]
            candidate_probs = [score / sum(candidate_scores) for score in candidate_scores]
            candidates.append(
                random.choices([j for j in range(population_size) if j not in candidates], weights=candidate_probs)[0])

        # Compute fitness scores for each election winner
        election_scores = [fitness_func(population[candidates[i]]) for i in range(num_candidates)]

        # Replace losing individuals with election winners
        for i in range(num_candidates):
            loser_index = fitness_scores.index(min(fitness_scores))
            population[loser_index] = population[candidates[i]]
            fitness_scores[loser_index] = election_scores[i]
        Convergence_curve[t] = fitness_scores
        t = t + 1
    # Return best individual from final population
    best_index = fitness_scores.index(max(fitness_scores))
    ct = time.time() - ct
    return population[best_index], Convergence_curve, fitness_scores, ct
