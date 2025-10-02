"""
Alpha/Omega Framework: Autonomous Prompt Optimization with Adaptive Evolution
Improved Version with Elitism, Curriculum Learning, and Independent Evaluation
"""

import os
import json
import time
import statistics
import random
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field

try:
    from openai import OpenAI
except ImportError:
    print("❌ Error: openai package not installed!")
    print("Install with: pip install openai")
    exit(1)


@dataclass
class PromptTemplate:
    """Represents a prompt strategy with associated metadata."""
    id: str
    template: str
    fitness_history: List[float] = field(default_factory=list)
    current_iteration_scores: List[float] = field(default_factory=list)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    is_elite: bool = False

    @property
    def average_fitness(self) -> float:
        """Use current iteration scores for selection, fall back to history."""
        if self.current_iteration_scores:
            return statistics.mean(self.current_iteration_scores)
        return statistics.mean(self.fitness_history) if self.fitness_history else 0.0

    @property
    def fitness_variance(self) -> float:
        return statistics.variance(self.fitness_history) if len(self.fitness_history) > 1 else 0.0


@dataclass
class Solution:
    """Represents a generated solution with evaluation metrics."""
    content: str
    prompt_id: str
    problem_id: str
    correctness: float = 0.0
    performance: float = 0.0
    quality: float = 0.0
    security: float = 0.0
    fitness: float = 0.0


@dataclass
class Problem:
    """Represents a problem to be solved."""
    id: str
    description: str
    test_cases: List[Dict] = field(default_factory=list)
    difficulty: str = "easy"
    difficulty_level: int = 1


class AdaptiveProblemSelector:
    """
    Adaptive Problem Selection Engine with Curriculum Learning.
    Gradually increases difficulty based on stable performance trends.
    """

    def __init__(self, problem_pool: List[Problem], 
                 pressure_threshold: float = 0.25,
                 stagnation_limit: int = 5):
        self.problem_pool = problem_pool
        self.current_problems = problem_pool.copy()
        self.pressure_threshold = pressure_threshold
        self.stagnation_limit = stagnation_limit
        self.fitness_history = []
        self.current_difficulty = 2  # Start at medium difficulty
        
    def check_stagnation_and_trigger_stress_test(self, 
                                                   diversity_pressure_index: float, 
                                                   stagnation_count: int) -> bool:
        """
        Check system stagnation and diversity pressure to trigger forced exploration.
        More conservative threshold to avoid excessive exploration.
        """
        trigger = (diversity_pressure_index > self.pressure_threshold and 
                   stagnation_count >= self.stagnation_limit)
        
        if trigger:
            print(f"🚨 STRESS TEST TRIGGERED: Pressure={diversity_pressure_index:.3f}, Stagnation={stagnation_count}")
        
        return trigger
    
    def adjust_difficulty_and_select_problems(self, current_avg_fitness: float) -> List[Problem]:
        """
        Gradually adjust problem difficulty based on stable performance trends.
        Uses 5-iteration window to avoid overreacting to noise.
        """
        self.fitness_history.append(current_avg_fitness)
        
        # Need sufficient history for stable trend analysis
        if len(self.fitness_history) < 10:
            return self._select_problems_by_difficulty(self.current_difficulty)
        
        # Compare recent 5 iterations vs previous 5
        recent_avg = statistics.mean(self.fitness_history[-5:])
        baseline_avg = statistics.mean(self.fitness_history[-10:-5])
        
        # Only adjust on clear, stable trends (±3% threshold)
        if recent_avg > baseline_avg + 0.03:
            new_difficulty = min(self.current_difficulty + 1, 5)
            if new_difficulty != self.current_difficulty:
                print(f"📈 Stable improvement detected -> Difficulty {self.current_difficulty} → {new_difficulty}")
                self.current_difficulty = new_difficulty
        elif recent_avg < baseline_avg - 0.03:
            new_difficulty = max(self.current_difficulty - 1, 1)
            if new_difficulty != self.current_difficulty:
                print(f"📉 Performance decline detected -> Difficulty {self.current_difficulty} → {new_difficulty}")
                self.current_difficulty = new_difficulty
        
        return self._select_problems_by_difficulty(self.current_difficulty)
    
    def _select_problems_by_difficulty(self, target_difficulty: int) -> List[Problem]:
        """Select problems centered around target difficulty."""
        selected = []
        
        for problem in self.problem_pool:
            # Higher probability for problems near target difficulty
            distance = abs(problem.difficulty_level - target_difficulty)
            prob = max(0.3, 1.0 - distance * 0.2)
            
            if random.random() < prob:
                selected.append(problem)
        
        # Ensure minimum problem set size
        if len(selected) < 3:
            selected = random.sample(self.problem_pool, min(5, len(self.problem_pool)))
        
        self.current_problems = selected
        return selected


class OmegaEngine:
    """Multi-dimensional evaluation engine with diversity pressure calculation."""

    def __init__(self, w_correctness=0.6, w_performance=0.15, w_quality=0.15, w_security=0.1):
        total = w_correctness + w_performance + w_quality + w_security
        self.weights = {
            'correctness': w_correctness / total,
            'performance': w_performance / total,
            'quality': w_quality / total,
            'security': w_security / total
        }

    def evaluate_solution(self, solution: Solution, problem: Problem) -> float:
        """Evaluate solution using fitness function."""
        solution.correctness = self._evaluate_correctness(solution, problem)
        solution.performance = self._evaluate_performance(solution, problem)
        solution.quality = self._evaluate_quality(solution, problem)
        solution.security = self._evaluate_security(solution, problem)

        solution.fitness = (
                self.weights['correctness'] * solution.correctness +
                self.weights['performance'] * solution.performance +
                self.weights['quality'] * solution.quality +
                self.weights['security'] * solution.security
        )

        return solution.fitness

    def calculate_diversity_pressure(self, solutions: List[Solution], avg_fitness: float) -> float:
        """
        Calculate diversity pressure index based on lower-performing group.
        Higher values indicate greater inefficiency and need for exploration.
        """
        if not solutions:
            return 0.0
        
        all_fitnesses = [s.fitness for s in solutions]
        all_fitnesses.sort()
        
        # Define lower 33% as underperforming group
        n_lower = max(1, len(all_fitnesses) // 3)
        lower_group_fitnesses = all_fitnesses[:n_lower]
        
        # Calculate pressure as average gap from mean
        pressure_sum = sum(max(0, avg_fitness - f) for f in lower_group_fitnesses)
        pressure_index = pressure_sum / n_lower if n_lower > 0 else 0.0
        
        return pressure_index

    def _evaluate_correctness(self, solution: Solution, problem: Problem) -> float:
        """Test functional correctness."""
        if not problem.test_cases:
            return 1.0

        passed = 0
        for test_case in problem.test_cases:
            try:
                local_ns = {}
                exec(solution.content, {}, local_ns)

                func = None
                for value in local_ns.values():
                    if callable(value):
                        func = value
                        break

                if func is None:
                    continue

                if 'input' in test_case:
                    result = func(test_case['input'])
                else:
                    result = func()

                if result == test_case['expected']:
                    passed += 1

            except Exception:
                continue

        return passed / len(problem.test_cases)

    def _evaluate_performance(self, solution: Solution, problem: Problem) -> float:
        """Evaluate code complexity as performance proxy."""
        lines = solution.content.count('\n')
        chars = len(solution.content)

        normalized_complexity = min(lines / 50, 1.0)
        normalized_size = min(chars / 500, 1.0)

        return 0.5 * (1 - normalized_complexity) + 0.5 * (1 - normalized_size)

    def _evaluate_quality(self, solution: Solution, problem: Problem) -> float:
        """Check code quality indicators."""
        score = 0.7

        if '"""' in solution.content or "'''" in solution.content:
            score += 0.1
        if '#' in solution.content:
            score += 0.05
        if 'def ' in solution.content:
            score += 0.1

        if len(solution.content) < 20:
            score -= 0.3
        if 'pass' == solution.content.strip().split('\n')[-1].strip():
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _evaluate_security(self, solution: Solution, problem: Problem) -> float:
        """Check for dangerous patterns."""
        score = 1.0
        dangerous = ['eval(', 'exec(', '__import__', 'os.system', 'subprocess']

        for pattern in dangerous:
            if pattern in solution.content:
                score -= 0.3

        return max(0.0, score)


class AlphaEngine:
    """Prompt generation and evolution engine with elitism and forced exploration."""

    def __init__(self, initial_prompts: List[str], mutation_rate=0.3, crossover_rate=0.2):
        self.prompts: Dict[str, PromptTemplate] = {}
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation_counter = 0

        for i, prompt_text in enumerate(initial_prompts):
            prompt_id = f"p0_{i}"
            self.prompts[prompt_id] = PromptTemplate(
                id=prompt_id,
                template=prompt_text,
                generation=0,
                is_elite=True  # Initial prompts are elite
            )

    def reset_iteration_scores(self):
        """Reset current iteration scores for all prompts."""
        for prompt in self.prompts.values():
            prompt.current_iteration_scores = []

    def generate_solution(self, problem: Problem, prompt_id: str, generator_func: Callable) -> Solution:
        """Generate solution using prompt and LLM."""
        prompt = self.prompts[prompt_id]
        full_prompt = prompt.template.replace("{problem}", problem.description)

        solution_content = generator_func(full_prompt)

        return Solution(
            content=solution_content,
            prompt_id=prompt_id,
            problem_id=problem.id
        )

    def evolve_prompts(self, top_k: int = 3, forced_exploration_mode: bool = False) -> None:
        """
        Evolve prompts with elitism preservation and optional forced exploration.
        
        Elitism: Top 20% of prompts are always preserved
        Forced Exploration: Uses lower-performing prompts for crossover
        """
        self.generation_counter += 1

        sorted_prompts = sorted(
            self.prompts.values(),
            key=lambda p: p.average_fitness,
            reverse=True
        )

        if not sorted_prompts:
            return

        # ELITISM: Preserve top 20%
        num_elites = max(2, len(sorted_prompts) // 5)
        elites = sorted_prompts[:num_elites]
        
        # Mark elite status
        for prompt in self.prompts.values():
            prompt.is_elite = False
        for elite in elites:
            elite.is_elite = True
        
        print(f"🛡️  Elites protected: {num_elites}/{len(self.prompts)} prompts")

        if forced_exploration_mode:
            print("🚨 FORCED EXPLORATION: Using lower-performing prompts for diversity")
            
            # Select parents from non-elite lower half
            non_elites = sorted_prompts[num_elites:]
            if len(non_elites) >= 4:
                lower_half_start = len(non_elites) // 2
                lower_half = non_elites[lower_half_start:]
                parents = random.sample(lower_half, min(top_k, len(lower_half)))
            else:
                parents = non_elites[:top_k] if non_elites else elites[:top_k]
            
            effective_mutation_rate = self.mutation_rate * 2.0
            effective_crossover_rate = self.crossover_rate * 1.5
            
        else:
            # Standard evolution: use top performers (but exclude pure elites to encourage diversity)
            parents = sorted_prompts[:top_k]
            effective_mutation_rate = self.mutation_rate
            effective_crossover_rate = self.crossover_rate

        new_prompts = []

        # Mutation
        if random.random() < effective_mutation_rate:
            for parent in parents[:2]:
                mutated = self._mutate_prompt(parent, forced_exploration_mode)
                if mutated:
                    new_prompts.append(mutated)

        # Crossover
        if random.random() < effective_crossover_rate and len(parents) >= 2:
            crossed = self._crossover_prompts(parents[0], parents[1])
            if crossed:
                new_prompts.append(crossed)

        for prompt in new_prompts:
            self.prompts[prompt.id] = prompt

    def _mutate_prompt(self, parent: PromptTemplate, radical: bool = False) -> Optional[PromptTemplate]:
        """Create mutated prompt variant."""
        standard_mutations = [
            lambda p: p + "\nBe concise and clear.",
            lambda p: p + "\nThink step by step.",
            lambda p: p + "\nConsider edge cases carefully.",
            lambda p: "Important: " + p,
            lambda p: p.replace("Write", "Implement"),
            lambda p: p + "\nUse efficient algorithms.",
        ]
        
        radical_mutations = [
            lambda p: p + "\nUse unconventional approaches.",
            lambda p: p + "\nChallenge standard assumptions.",
            lambda p: p.replace("function", "creative solution"),
            lambda p: "Experimental: " + p + "\nTry novel patterns.",
            lambda p: p + "\nPrioritize innovation over convention.",
        ]

        mutations = radical_mutations if radical else standard_mutations
        mutation_func = random.choice(mutations)
        new_template = mutation_func(parent.template)

        new_id = f"p{self.generation_counter}_m{random.randint(1000, 9999)}"

        return PromptTemplate(
            id=new_id,
            template=new_template,
            generation=self.generation_counter,
            parent_ids=[parent.id]
        )

    def _crossover_prompts(self, parent1: PromptTemplate, parent2: PromptTemplate) -> Optional[PromptTemplate]:
        """Combine elements from two prompts."""
        sentences1 = [s.strip() for s in parent1.template.split('.') if s.strip()]
        sentences2 = [s.strip() for s in parent2.template.split('.') if s.strip()]

        if not sentences1 or not sentences2:
            return None

        mid1 = len(sentences1) // 2
        mid2 = len(sentences2) // 2

        new_sentences = sentences1[:mid1] + sentences2[mid2:]
        new_template = '. '.join(new_sentences) + '.'

        new_id = f"p{self.generation_counter}_c{random.randint(1000, 9999)}"

        return PromptTemplate(
            id=new_id,
            template=new_template,
            generation=self.generation_counter,
            parent_ids=[parent1.id, parent2.id]
        )

    def update_fitness(self, prompt_id: str, fitness: float) -> None:
        """Update fitness for both current iteration and history."""
        if prompt_id in self.prompts:
            self.prompts[prompt_id].current_iteration_scores.append(fitness)
            self.prompts[prompt_id].fitness_history.append(fitness)

    def get_best_prompts(self, k: int = 5) -> List[PromptTemplate]:
        """Get top k prompts based on average fitness."""
        return sorted(
            self.prompts.values(),
            key=lambda p: statistics.mean(p.fitness_history) if p.fitness_history else 0.0,
            reverse=True
        )[:k]


class AlphaOmegaFramework:
    """Main framework with elitism, curriculum learning, and independent evaluation."""

    def __init__(self, initial_prompts: List[str], generator_func: Callable, 
                 problem_pool: List[Problem], omega_weights=None):
        self.alpha = AlphaEngine(initial_prompts)
        self.omega = OmegaEngine(**omega_weights) if omega_weights else OmegaEngine()
        self.delta = AdaptiveProblemSelector(problem_pool)
        self.generator_func = generator_func
        
        self.problems = problem_pool.copy()
        self.evolution_history = []
        self.iteration_count = 0
        self.max_fitness_history = []
        self.stagnation_count = 0

    def run_evolution_cycle(self, iterations=20, solutions_per_prompt=1) -> Dict:
        """Run complete evolution cycle with improved mechanisms."""
        print(f"🚀 Starting Improved Alpha/Omega/Delta Evolution")
        print(f"📊 Initial prompts: {len(self.alpha.prompts)}")
        print(f"🔢 Problem pool: {len(self.delta.problem_pool)}")
        print(f"🔄 Solutions per prompt: {solutions_per_prompt}")
        print(f"🛡️  Elitism: Enabled (top 20%)")
        print(f"📚 Curriculum Learning: Enabled")
        print(f"🎯 Independent Evaluation: Enabled")
        print("-" * 70)

        for iteration in range(iterations):
            self.iteration_count = iteration
            iteration_start = time.time()

            # Run iteration with all improvements
            iteration_data = self._run_iteration(iteration, solutions_per_prompt)

            # Track history
            self.evolution_history.append(iteration_data)

            # Progress report
            if iteration % 5 == 0 or iteration_data.get('forced_exploration', False):
                elite_count = sum(1 for p in self.alpha.prompts.values() if p.is_elite)
                print(f"Iter {iteration:2d} | Avg: {iteration_data['avg_fitness']:.3f} | "
                      f"Max: {iteration_data['max_fitness']:.3f} | "
                      f"Pressure: {iteration_data['diversity_pressure']:.3f} | "
                      f"Prompts: {len(self.alpha.prompts):2d} (Elites: {elite_count}) | "
                      f"Diff: {self.delta.current_difficulty} | "
                      f"Time: {iteration_data['duration']:.1f}s")

        print("-" * 70)
        print("✅ Evolution complete!")

        return self.get_statistics()

    def _run_iteration(self, iteration: int, solutions_per_prompt: int) -> Dict:
        """Run single iteration with independent evaluation and all improvements."""
        iteration_start = time.time()
        
        # Reset iteration scores for independent evaluation
        self.alpha.reset_iteration_scores()
        
        all_solutions = []
        iteration_fitness = []

        # 1. GENERATION & EVALUATION
        for problem in self.problems:
            for prompt_id in list(self.alpha.prompts.keys()):
                for _ in range(solutions_per_prompt):
                    try:
                        solution = self.alpha.generate_solution(
                            problem, prompt_id, self.generator_func
                        )
                        fitness = self.omega.evaluate_solution(solution, problem)
                        
                        self.alpha.update_fitness(prompt_id, fitness)
                        iteration_fitness.append(fitness)
                        all_solutions.append(solution)

                    except Exception as e:
                        print(f"⚠️  Error in iteration {iteration}: {e}")
                        continue

        # 2. CALCULATE METRICS
        avg_fitness = statistics.mean(iteration_fitness) if iteration_fitness else 0.0
        max_fitness = max(iteration_fitness) if iteration_fitness else 0.0
        
        # 3. DIVERSITY PRESSURE INDEX
        diversity_pressure = self.omega.calculate_diversity_pressure(all_solutions, avg_fitness)

        # 4. STAGNATION DETECTION
        self.max_fitness_history.append(max_fitness)
        if len(self.max_fitness_history) >= 2:
            if abs(self.max_fitness_history[-1] - self.max_fitness_history[-2]) < 0.01:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
        
        # 5. ADAPTIVE PROBLEM SELECTION (Curriculum Learning)
        forced_exploration = self.delta.check_stagnation_and_trigger_stress_test(
            diversity_pressure, self.stagnation_count
        )
        self.problems = self.delta.adjust_difficulty_and_select_problems(avg_fitness)

        # 6. EVOLUTION (every 5 iterations or when forced)
        if (iteration > 0 and iteration % 5 == 0) or forced_exploration:
            self.alpha.evolve_prompts(top_k=3, forced_exploration_mode=forced_exploration)
            
            if forced_exploration:
                self.stagnation_count = 0

        return {
            'iteration': iteration,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'diversity_pressure': diversity_pressure,
            'stagnation_count': self.stagnation_count,
            'num_prompts': len(self.alpha.prompts),
            'num_elites': sum(1 for p in self.alpha.prompts.values() if p.is_elite),
            'num_problems': len(self.problems),
            'current_difficulty': self.delta.current_difficulty,
            'forced_exploration': forced_exploration,
            'duration': time.time() - iteration_start
        }

    def get_statistics(self) -> Dict:
        """Get evolution statistics."""
        if not self.evolution_history:
            return {}

        fitness_values = [h['avg_fitness'] for h in self.evolution_history]
        best_prompts = self.alpha.get_best_prompts(k=5)

        return {
            'total_iterations': len(self.evolution_history),
            'final_avg_fitness': fitness_values[-1],
            'initial_avg_fitness': fitness_values[0],
            'improvement': fitness_values[-1] - fitness_values[0],
            'improvement_percent': ((fitness_values[-1] - fitness_values[0]) / fitness_values[0] * 100) if fitness_values[0] > 0 else 0,
            'max_fitness_achieved': max(h['max_fitness'] for h in self.evolution_history),
            'total_prompts_created': len(self.alpha.prompts),
            'forced_exploration_count': sum(1 for h in self.evolution_history if h.get('forced_exploration', False)),
            'avg_diversity_pressure': statistics.mean(h['diversity_pressure'] for h in self.evolution_history),
            'best_prompts': [
                {
                    'id': p.id,
                    'fitness': statistics.mean(p.fitness_history) if p.fitness_history else 0.0,
                    'generation': p.generation,
                    'evaluations': len(p.fitness_history),
                    'is_elite': p.is_elite,
                    'template': p.template
                }
                for p in best_prompts
            ],
            'evolution_history': self.evolution_history
        }

    def get_best_prompt(self) -> PromptTemplate:
        """Get best prompt."""
        best = self.alpha.get_best_prompts(k=1)
        return best[0] if best else None


def create_openai_generator(model="gpt-4o-mini", temperature=0.7):
    """Create OpenAI generator function."""
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable not set!")

    client = OpenAI(api_key=api_key)

    def generator(prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert Python programmer. Generate only the Python function code without explanations or markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=400
            )

            code = response.choices[0].message.content.strip()

            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()

            return code

        except Exception as e:
            print(f"❌ Error generating solution: {e}")
            return "def solution():\n    pass"

    return generator


def main():
    """Main execution function."""
    print("=" * 70)
    print("🧬 Improved Alpha/Omega/Delta Framework")
    print("Elitism + Curriculum Learning + Independent Evaluation")
    print("=" * 70)
    print()

    print("🔑 Checking API key...")
    if not os.environ.get("API_KEY"):
        print("❌ Error: API_KEY environment variable not found!")
        print("Set it with: export API_KEY='your-openai-api-key'")
        return
    print("✅ API key found!")
    print()

    # Initial prompts
    initial_prompts = [
        "Write a Python function to solve: {problem}\nReturn only the function code.",

        "Problem: {problem}\n\nWrite a clean Python function that:\n1. Handles the input correctly\n2. Returns the expected output\n3. Is efficient and readable",

        "Implement a solution for: {problem}\n\nRequirements:\n- Write concise, pythonic code\n- Include a docstring\n- Handle edge cases\n- Optimize for clarity",

        "Task: {problem}\n\nThink step by step:\n1. Understand the problem\n2. Plan your approach\n3. Implement the solution\n4. Test edge cases\n\nProvide only the Python function.",

        "{problem}\n\nWrite an optimized Python function. Focus on:\n- Time complexity\n- Space efficiency\n- Clean code structure",
    ]

    # Problem pool with difficulty levels
    problem_pool = [
        Problem(
            id="reverse_string",
            description="Write a function 'reverse_string(s)' that takes a string and returns it reversed.",
            difficulty="easy",
            difficulty_level=1,
            test_cases=[
                {'input': 'hello', 'expected': 'olleh'},
                {'input': 'world', 'expected': 'dlrow'},
                {'input': 'a', 'expected': 'a'},
                {'input': '', 'expected': ''},
            ]
        ),
        Problem(
            id="sum_list",
            description="Write a function 'sum_list(numbers)' that takes a list of integers and returns their sum.",
            difficulty="easy",
            difficulty_level=1,
            test_cases=[
                {'input': [1, 2, 3], 'expected': 6},
                {'input': [10, 20, 30], 'expected': 60},
                {'input': [], 'expected': 0},
                {'input': [-1, 1], 'expected': 0},
            ]
        ),
        Problem(
            id="find_max",
            description="Write a function 'find_max(numbers)' that returns the maximum number in a list.",
            difficulty="medium",
            difficulty_level=2,
            test_cases=[
                {'input': [1, 5, 3, 9, 2], 'expected': 9},
                {'input': [-5, -1, -10], 'expected': -1},
                {'input': [42], 'expected': 42},
            ]
        ),
        Problem(
            id="count_vowels",
            description="Write a function 'count_vowels(text)' that counts vowels (a,e,i,o,u) in a string (case-insensitive).",
            difficulty="medium",
            difficulty_level=3,
            test_cases=[
                {'input': 'hello', 'expected': 2},
                {'input': 'aeiou', 'expected': 5},
                {'input': 'xyz', 'expected': 0},
                {'input': 'HELLO', 'expected': 2},
            ]
        ),
        Problem(
            id="is_palindrome",
            description="Write a function 'is_palindrome(s)' that returns True if string is a palindrome, False otherwise.",
            difficulty="medium",
            difficulty_level=3,
            test_cases=[
                {'input': 'racecar', 'expected': True},
                {'input': 'hello', 'expected': False},
                {'input': 'a', 'expected': True},
                {'input': '', 'expected': True},
            ]
        ),
        Problem(
            id="fibonacci",
            description="Write a function 'fibonacci(n)' that returns the nth Fibonacci number (0-indexed).",
            difficulty="hard",
            difficulty_level=4,
            test_cases=[
                {'input': 0, 'expected': 0},
                {'input': 1, 'expected': 1},
                {'input': 5, 'expected': 5},
                {'input': 10, 'expected': 55},
            ]
        ),
        Problem(
            id="merge_sorted",
            description="Write a function 'merge_sorted(list1, list2)' that merges two sorted lists into one sorted list.",
            difficulty="hard",
            difficulty_level=5,
            test_cases=[
                {'input': ([1, 3, 5], [2, 4, 6]), 'expected': [1, 2, 3, 4, 5, 6]},
                {'input': ([1], [2]), 'expected': [1, 2]},
                {'input': ([], [1, 2]), 'expected': [1, 2]},
            ]
        ),
    ]

    print("🤖 Initializing OpenAI generator (gpt-4o-mini)...")
    generator = create_openai_generator(model="gpt-4o-mini", temperature=0.7)
    print("✅ Generator ready!")
    print()

    print("🏗️  Initializing Improved Framework...")
    framework = AlphaOmegaFramework(
        initial_prompts=initial_prompts,
        generator_func=generator,
        problem_pool=problem_pool,
        omega_weights={
            'w_correctness': 0.6,
            'w_performance': 0.15,
            'w_quality': 0.15,
            'w_security': 0.1
        }
    )
    print("✅ Framework initialized!")
    print()

    # Run evolution
    print("=" * 70)
    print("🔬 STARTING IMPROVED EVOLUTION EXPERIMENT")
    print("=" * 70)
    stats = framework.run_evolution_cycle(
        iterations=20,
        solutions_per_prompt=1
    )

    # Display results
    print()
    print("=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"Total Iterations: {stats['total_iterations']}")
    print(f"Initial Avg Fitness: {stats['initial_avg_fitness']:.3f}")
    print(f"Final Avg Fitness: {stats['final_avg_fitness']:.3f}")
    print(f"Improvement: {stats['improvement']:+.3f} ({stats['improvement_percent']:+.1f}%)")
    print(f"Max Fitness: {stats['max_fitness_achieved']:.3f}")
    print(f"Total Prompts: {stats['total_prompts_created']}")
    print(f"Forced Explorations: {stats['forced_exploration_count']}")
    print(f"Avg Diversity Pressure: {stats['avg_diversity_pressure']:.3f}")

    print()
    print("=" * 70)
    print("🏆 TOP 5 EVOLVED PROMPTS")
    print("=" * 70)
    for i, p in enumerate(stats['best_prompts'][:5], 1):
        elite_marker = "👑" if p['is_elite'] else "  "
        print(f"\n{elite_marker} #{i} [Gen {p['generation']}] Fitness: {p['fitness']:.3f} ({p['evaluations']} evals)")
        print(f"ID: {p['id']}")
        print(f"Template:\n{p['template']}")
        print("-" * 70)

    # Save results
    print()
    print("💾 Saving results...")
    with open("alpha_omega_delta_improved_results.json", "w", encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("✅ Results saved to: alpha_omega_delta_improved_results.json")

    best = framework.get_best_prompt()
    if best:
        print()
        print("=" * 70)
        print("⭐ CHAMPION PROMPT")
        print("=" * 70)
        elite_status = "👑 ELITE" if best.is_elite else "Standard"
        print(f"Status: {elite_status}")
        print(f"Fitness: {statistics.mean(best.fitness_history) if best.fitness_history else 0:.3f}")
        print(f"Generation: {best.generation}")
        print(f"Total Evaluations: {len(best.fitness_history)}")
        if len(best.fitness_history) > 1:
            print(f"Std Dev: {statistics.stdev(best.fitness_history):.3f}")
        print()
        print(best.template)

    print()
    print("=" * 70)
    print("✨ Improved Evolution Experiment Complete!")
    print("=" * 70)
    print()
    print("Key Improvements Applied:")
    print("✅ Elitism: Top 20% prompts preserved")
    print("✅ Curriculum Learning: Gradual difficulty adjustment")
    print("✅ Independent Evaluation: Fresh scores each iteration")
    print("✅ Conservative Exploration: Higher thresholds for forced mode")


if __name__ == "__main__":
    main()