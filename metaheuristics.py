import numpy as np
from collections.abc import Iterable

class GeneticAlgorithm:
    def __init__(
            self,
            baseline,
            pop_size=100,
            init_method="random",
            k_best_individuals=0.1,
            n_parents=2,
            breeding_method="average",
            individual_mutation_rate=1.0,
            gene_mutation_rate=1.0,
            mutation_intensity=1.0,
            constraints=None,
    ):
        self.baseline = baseline
        self.init_method = init_method
        self.pop_size = pop_size
        self.k = round(k_best_individuals if k_best_individuals > 1 else pop_size*k_best_individuals)
        self.n_parents = round(n_parents if n_parents > 1 else self.k*n_parents)
        self.breeding = breeding_method
        self.mut_ratio = individual_mutation_rate
        self.mut_rate = gene_mutation_rate
        self.mut_intensity = mutation_intensity
        self.constraints = constraints
        self.constrained = True
        try:
            self.constraints[0]
        except:
            self.constrained = False

        self.initialize()

    def initialize(self):
        self.population = []
        self.total_generations = 0
        if self.init_method=="base":
            self.population = [self.baseline]*self.pop_size
            self.mutate(1, 1, 1)
        elif self.init_method=="random":
            for p in range(self.pop_size):
                if self.constrained:
                    individual = [np.random.uniform(self.constraints[g][0], self.constraints[g][1]) for g in range(self.baseline.shape[0])]
                else:
                    individual = np.random.normal(size=self.baseline.shape)
                self.population.append(individual)
        self.population = np.array(self.population)
        self.history = {"solution" : [], "score" : []}
    
    def reset_mutation(self, gens):
        self.mutation_ratio_range = None
        self.mutation_rate_range = None
        self.mutation_intensity_range = None
        if type(self.mut_ratio) != float:
            self.mutation_ratio_range = np.linspace(self.mut_ratio[0], self.mut_ratio[1], gens)
            self.mutation_ratio = self.mutation_ratio_range[0]
        else:
            self.mutation_ratio = self.mut_ratio
        if type(self.mut_rate) != float:
            self.mutation_rate_range = np.linspace(self.mut_rate[0], self.mut_rate[1], gens)
            self.mutation_rate = self.mutation_rate_range[0]
        else:
            self.mutation_rate = self.mut_rate
        if type(self.mut_intensity) != float:
            self.mutation_intensity_range = np.linspace(self.mut_intensity[0], self.mut_intensity[1], gens)
            self.mutation_intensity = self.mutation_intensity_range[0]
        else:
            self.mutation_intensity = self.mut_intensity
    
    def breed(self, k_best, k_best_fitness):
        new_population = []
        for p in range(self.pop_size):
            parent_indices = np.random.choice(self.k, self.n_parents, replace=False)
            if self.breeding=="random":
                weights = np.random.uniform(size=self.n_parents)
            if self.breeding=="average":
                weights = np.ones(shape=self.n_parents)
            if self.breeding=="fitness":
                weights = k_best_fitness[parent_indices]
            weights = weights/weights.sum()
            parents = k_best[parent_indices]
            individual = parents.T @ weights
            new_population.append(individual)
        self.population = np.array(new_population)

    def mutate(self, ratio, gene_rate, intensity):
        individuals = np.random.choice(self.pop_size, int(ratio*self.pop_size), replace=False)
        for i in individuals:
            genes = np.random.choice(len(self.population[i]), int(gene_rate*len(self.population[i])), replace=False)
            for g in genes:
                if self.constrained:
                    lower = np.random.uniform(self.constraints[g][0]-self.population[i][g], 0)
                    higher = np.random.uniform(0, self.constraints[g][1]-self.population[i][g])
                    amount = np.random.choice([lower, higher]) * intensity
                else:
                    amount = np.random.uniform(-1,1) * self.population[i][g] * intensity
                self.population[i][g] += amount

    def run(self, eval_func, max_generations=None, score_threshold=None, patience=None, tolerance=None, re_init=True, verbose=False):
        #Setup stopping criteria
        stopping_criteria = []
        if not (max_generations or score_threshold or tolerance or patience):
            max_generations = 50
            patience = 10
        if max_generations:
            stopping_criteria.append("gen")
        if score_threshold:
            stopping_criteria.append("score")
        if patience:
            if not tolerance: tolerance = 0
            stopping_criteria.append("early")

        #Initialize
        if re_init:
            self.initialize()
        self.reset_mutation(max_generations)
        
        self.best_score = -np.inf
        self.best_solution = None
        self.best_iter = None

        #Evolve
        current_generation = 0
        waits = 0
        while True:
            #Evaluate
            scores = []
            for ind in self.population:
                fitness = eval_func(ind)
                scores.append(fitness)
            scores = np.array(scores)
            k_best_indices = scores.argsort()[::-1][:self.k]
            k_best_scores = scores[k_best_indices]
            k_best_individuals = self.population[k_best_indices]
            best_score = k_best_indices[0]
            self.history["solution"].append(np.copy(self.population[best_score]))
            self.history["score"].append(scores[best_score])
            improvement = scores[best_score] - self.best_score
            if scores[best_score] > self.best_score:
                self.best_score = scores[best_score]
                self.best_solution = self.population[best_score]
                self.best_iter = self.total_generations
            if verbose: print(f"Gen: {self.total_generations}, Best score: {scores[best_score]}")
            
            #Stopping criteria
            if "gen" in stopping_criteria:
                if current_generation >= max_generations: break
            if "score" in stopping_criteria:
                if self.best_score >= score_threshold: break
            if "early" in stopping_criteria:
                if improvement <= tolerance:
                    waits += 1
                    if waits > patience: break
                else:
                    waits = 0
            
            #Breeding and mutation
            self.breed(k_best_individuals, k_best_scores)
            if self.mutation_ratio_range: self.mutation_ratio = self.mutation_ratio_range[current_generation]
            if self.mutation_rate_range: self.mutation_rate = self.mutation_rate_range[current_generation]
            if self.mutation_intensity_range: self.mutation_intensity = self.mutation_intensity_range[current_generation]
            self.mutate(self.mutation_ratio, self.mutation_rate, self.mutation_intensity)

            current_generation += 1
            self.total_generations += 1
    
    def get_best_solution(self):
        return self.history["solution"][np.argmax(np.array(self.history["score"]))]

class PSO:
    def __init__(
            self,
            baseline,
            swarm_size=100,
            init_method="random",
            inertia=0.8,
            cogn_coeff=0.1,
            soc_coeff=0.1,
            velocity_magnitude=1,
            max_velocity=None,
            constraints=None
    ):
        self.baseline = baseline
        self.init_method = init_method
        self.swarm_size = swarm_size
        self.inert = inertia
        self.cogn = cogn_coeff
        self.soc = soc_coeff
        self.vel_magn = velocity_magnitude
        self.max_vel = max_velocity
        self.constraints = constraints
        self.constrained = True
        try:
            self.constraints[0]
        except:
            self.constrained = False
        self.initialize()
        
    def initialize(self):
        self.particles = []
        self.velocities = []
        self.total_iters = 0
        if self.init_method=="base":
            self.particles = [self.baseline]*self.swarm_size
        elif self.init_method=="random":
            for p in range(self.swarm_size):
                if self.constrained:
                    individual = [np.random.uniform(self.constraints[g][0], self.constraints[g][1]) for g in range(self.baseline.shape[0])]
                    vel = np.array([np.random.uniform(self.constraints[g][0], self.constraints[g][1]) for g in range(self.baseline.shape[0])]) - np.array(individual)
                else:
                    individual = np.random.normal(size=self.baseline.shape)
                    vel = np.random.normal(size=self.baseline.shape) - individual
                self.particles.append(individual)
                self.velocities.append(vel)
        self.particles = np.array(self.particles)
        self.velocities = np.array(self.velocities)
        self.best_scores = np.zeros(self.particles.shape[0]) - np.inf
        self.best_positions = np.copy(self.particles)
        self.history = {"solution" : [], "score" : []}
    
    def reset_coefficients(self, iter):
        self.inertia_range = None
        self.cogn_coeff_range = None
        self.soc_coeff_range = None
        if isinstance(self.inert, Iterable):
            self.inertia_range = np.linspace(self.inert[0], self.inert[1], iter)
            self.inertia = self.inertia_range[0]
        else:
            self.inertia = self.inert
        if isinstance(self.cogn, Iterable):
            self.cogn_coeff_range = np.linspace(self.cogn[0], self.cogn[1], iter)
            self.cogn_coeff = self.cogn_coeff_range[0]
        else:
            self.cogn_coeff = self.cogn
        if isinstance(self.soc, Iterable):
            self.soc_coeff_range = np.linspace(self.soc[0], self.soc[1], iter)
            self.soc_coeff = self.soc_coeff_range[0]
        else:
            self.soc_coeff = self.soc

    def update_velocity(self, inertia, cogn_coeff, soc_coeff):
        cogn_dir = self.best_positions-self.particles
        soc_dir = self.best_solution-self.particles
        self.velocities = inertia*self.velocities + cogn_coeff*np.random.uniform()*(cogn_dir) + soc_coeff*np.random.uniform()*(soc_dir)
        if self.max_vel:
            self.velocities = np.clip(self.velocities, -self.max_vel, self.max_vel)
        self.particles += self.velocities*self.vel_magn
        if self.constrained:
            for i,p in enumerate(self.particles):
                for j,s in enumerate(p):
                    if s < self.constraints[j][0]: self.particles[i][j] = self.constraints[j][0]
                    if s > self.constraints[j][1]: self.particles[i][j] = self.constraints[j][1]
    
    def run(self, eval_func, max_iter=None, score_threshold=None, patience=None, tolerance=None, re_init=True, verbose=False):
        #Setup stopping criteria
        stopping_criteria = []
        if not (max_iter or score_threshold or tolerance or patience):
            max_iter = 50
            patience = 10
        if max_iter:
            stopping_criteria.append("iter")
        if score_threshold:
            stopping_criteria.append("score")
        if patience:
            if not tolerance: tolerance = 0
            stopping_criteria.append("early")

        #Initialize
        if re_init:
            self.initialize()
        self.reset_coefficients(max_iter)
        
        self.best_score = -np.inf
        self.best_solution = None
        self.best_iter = None

        #Optimize
        current_iter = 0
        waits = 0
        while True:
            #Evaluate
            scores = []
            for i,ind in enumerate(self.particles):
                score = eval_func(ind)
                if score > self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = np.copy(ind)
                scores.append(score)
            scores = np.array(scores)
            best_score = scores.argsort()[::-1][0]
            self.history["solution"].append(np.copy(self.particles[best_score]))
            self.history["score"].append(scores[best_score])
            improvement = scores[best_score] - self.best_score
            if scores[best_score] > self.best_score:
                self.best_score = scores[best_score]
                self.best_solution = np.copy(self.particles[best_score])
                self.best_iter = self.total_iters
            if verbose: print(f"Iter: {self.total_iters}, Best score: {scores[best_score]}")
            
            #Stopping criteria
            if "iter" in stopping_criteria:
                if current_iter >= max_iter: break
            if "score" in stopping_criteria:
                if self.best_score >= score_threshold: break
            if "early" in stopping_criteria:
                if improvement <= tolerance:
                    waits += 1
                    if waits > patience: break
                else:
                    waits = 0
            
            #Movement and velocity update
            if isinstance(self.inert, Iterable): self.inertia = self.inertia_range[current_iter]
            if isinstance(self.cogn, Iterable): self.cogn_coeff = self.cogn_coeff_range[current_iter]
            if isinstance(self.soc, Iterable): self.soc_coeff = self.soc_coeff_range[current_iter]
            self.update_velocity(self.inertia, self.cogn_coeff, self.soc_coeff)

            current_iter += 1
            self.total_iters += 1
    
    def get_best_solution(self):
        return self.history["solution"][np.argmax(np.array(self.history["score"]))]

# Angle Modulation Particle Swarm Optimization
class AMPSO(PSO):
    def __init__(
            self, 
            num_bits, 
            swarm_size=100,
            inertia=0.8,
            cogn_coeff=0.1,
            soc_coeff=0.1,
            velocity_magnitude=1,
            constraints=None,
            use_amplitude=True,
            use_minmax=True,
        ):
        self.num_bits = num_bits
        self.num_vars = 4
        if use_amplitude: self.num_vars += 1
        if use_minmax: self.num_vars += 2
        baseline = np.random.uniform(size=self.num_vars)
        super().__init__(
            baseline=baseline,
            swarm_size=swarm_size,
            inertia=inertia,
            cogn_coeff=cogn_coeff,
            soc_coeff=soc_coeff,
            velocity_magnitude=velocity_magnitude,
            constraints=constraints)
    
    def generate_bitstring_decorator(self, func):
        def inner(particle, *args, **kwargs):
            a, b, c, d = particle[:4]
            amplitude = 1
            min_val = max_val = None
            if self.num_vars == 7:
                amplitude, min_val, max_val = particle[4], min(particle[5:]), max(particle[5:])
            elif self.num_vars == 6:
                min_val, max_val = min(particle[4:]), max(particle[4:])
            elif self.num_vars == 5:
                amplitude = particle[4]
            
            x = np.linspace(min_val, max_val, self.num_bits) if min_val and max_val else np.arange(0, self.num_bits, dtype=np.float32)
            y = amplitude*np.sin(2*np.pi*(x-a) * b*np.cos(2*np.pi*c*(x-a))) + d
            bits = np.where(y>0, 1, 0)
            return func(bits, *args, **kwargs)
        return inner
    
    def get_solution(self, particle):
        a, b, c, d = particle[:4]
        amplitude = 1
        min_val = max_val = None
        if self.num_vars == 7:
            amplitude, min_val, max_val = particle[4], min(particle[5:]), max(particle[5:])
        elif self.num_vars == 6:
            min_val, max_val = min(particle[4:]), max(particle[4:])
        elif self.num_vars == 5:
            amplitude = particle[4]

        x = np.linspace(min_val, max_val, self.num_bits) if min_val and max_val else np.arange(0, self.num_bits, dtype=np.float32)
        y = amplitude*np.sin(2*np.pi*(x-a) * b*np.cos(2*np.pi*c*(x-a))) + d
        bits = np.where(y>0, 1, 0)
        return bits

    def get_best_solution(self):
        return self.get_solution(self.best_solution)
    
    def run(self, eval_func, max_iter=None, score_threshold=None, patience=None, tolerance=None, re_init=True, verbose=False):
        eval_func = self.generate_bitstring_decorator(eval_func)
        super().run(eval_func, max_iter, score_threshold, patience, tolerance, re_init, verbose)

if __name__ == "__main__":
    pass