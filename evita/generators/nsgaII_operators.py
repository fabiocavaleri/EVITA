import os
import random
from copy import deepcopy

import time
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.termination import Termination

from evita.config import ValidationConfig, ScenarioGeneratorConfig
from evita.generators import scenario_mutation
from evita.validator import ScenarioValidator, EVITAScenarioValidator
from evita.generators import utils as scenario_generation_utils

import matplotlib.pyplot as plt
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.planning.planning_problem import PlanningProblemSet
from conversion_scripts.utils.conversion_utils import generate_traffic_lights


class ScenarioSampling(Sampling):

    def __init__(self, base_scenario, base_planning_problem_set, scenario_generator_configuration):
        super().__init__()
        self.base_scenario = base_scenario
        self.base_planning_problem_set = base_planning_problem_set
        self.scenario_generator_configuration = scenario_generator_configuration
        self.scenario_validator_configuration = ValidationConfig()
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration)
        self.add_random_planning_problem = scenario_mutation.add_random_planning_problem

        self.scenario_counter = 0

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 2), None, dtype=object)

        for i in range(n_samples):
            individual = self._generate_random_individual()
            X[i, 0] = individual[0]
            X[i, 1] = individual[1]
            
        return X

    def _generate_random_individual(self):
        """
        First set a target number of AVs. Then try as much as possible to generate a valid scenario with them. If this fails, decrease the target number of AV and repeat.
        # TODO Refactor this and ensure the same code is used for random and 1+1EA
        """

        target_number_of_planning_problems = random.randint(self.scenario_generator_configuration.min_amount_pp,
                                                            self.scenario_generator_configuration.max_amount_pp)

        while target_number_of_planning_problems >= self.scenario_generator_configuration.min_amount_pp:
            print(f"Try to generate a scenario with {target_number_of_planning_problems}")
            
            for _ in range(0, self.scenario_validator_configuration.retry_generation):
                
                try:
                    self.scenario_counter += 1

                    _scenario = deepcopy(self.base_scenario)
                    _planning_problem_set = deepcopy(self.base_planning_problem_set)

                    for _ in range(0, target_number_of_planning_problems):
                        _planning_problem_set = self.add_random_planning_problem(self.scenario_generator_configuration,
                                                                                _scenario, _planning_problem_set)
                    # is_scenario_valid returns a tuple
                    if self.scenario_validator.is_scenario_valid(_scenario, _planning_problem_set)[0]:
                        
                        print(f"Scenario {self.scenario_counter} generated with {target_number_of_planning_problems}")
                        setattr(_scenario, "debug_id", self.scenario_counter)

                        return _scenario, _planning_problem_set
                except Exception as exc:
                    print(f"An exception occurred while generating the scenario. {exc}")
                    pass

            target_number_of_planning_problems = target_number_of_planning_problems - 1

        raise Exception("Cannot generate any scenario!")

class EVITAScenarioSampling(ScenarioSampling):
    def __init__(self, base_scenario, base_planning_problem_set, scenario_generator_configuration, map_file_path):
        super().__init__(base_scenario, base_planning_problem_set, scenario_generator_configuration)
        self.scenario_validator = EVITAScenarioValidator(self.scenario_validator_configuration)
        self.add_random_planning_problem = scenario_mutation.add_evita_random_planning_problem

        self.map_file_path = map_file_path

    def _do(self, problem, n_samples, **kwargs):
        self.base_scenario = generate_traffic_lights(self.base_scenario, self.scenario_generator_configuration.max_time_step, self.map_file_path)

        X = np.full((n_samples, 2), None, dtype=object)

        print("\n")

        # Generate `n_samples` scenarios and planning problems
        for i in range(n_samples):
            individual = self._generate_random_individual()
            X[i, 0] = individual[0]
            X[i, 1] = individual[1]

        return X

    def _generate_random_individual(self):
        """
        First set a target number of AVs. 
        Then try as much as possible to generate a valid scenario with them. 
        If this fails, decrease the target number of AV and repeat.
        """

        # Randomly choose the number of AVs (planning problems) to generate
        target_number_of_planning_problems = random.randint(self.scenario_generator_configuration.min_amount_pp,
                                                            self.scenario_generator_configuration.max_amount_pp)

        while target_number_of_planning_problems >= self.scenario_generator_configuration.min_amount_pp:
            print(f"Try to generate a scenario with {target_number_of_planning_problems}")
            
            for _ in range(0, self.scenario_validator_configuration.retry_generation):
                try:
                    _scenario = deepcopy(self.base_scenario)
                    _planning_problem_set = deepcopy(self.base_planning_problem_set)

                    # Generate `target_number_of_planning_problems` planning problems for the scenario
                    for _ in range(0, target_number_of_planning_problems):
                        _planning_problem_set = self.add_random_planning_problem(self.scenario_generator_configuration,
                                                                                    _scenario, _planning_problem_set, self.map_file_path)
                    
                    # Check if the generated scenario and planning problems are valid
                    if self.scenario_validator.is_scenario_valid(_scenario, _planning_problem_set)[0]:
                        print(f"\nScenario {self.scenario_counter} generated with {target_number_of_planning_problems}")
                        setattr(_scenario, "debug_id", self.scenario_counter)

                        # Print the main characteristics of the generated planning problems
                        for pp in _planning_problem_set._planning_problem_dict.values():
                            print(f"Planning Problem: {str(pp.planning_problem_id):<10} "
                                  f"Initial Lane: {str(_scenario.lanelet_network.find_lanelet_by_position([pp.initial_state.position])):<20} "
                                  f"Goal Lane: {str(_scenario.lanelet_network.find_lanelet_by_position([pp.goal.state_list[-1].position.center])):<20} "
                                  f"Route: {pp.path}")

                        print(f"\n{'=' * 150}\n")

                        self.scenario_counter += 1

                        return _scenario, _planning_problem_set
                except Exception as exc:
                    print(f"An exception occurred while generating the scenario. {exc}")
                    pass

class ScenarioMutation(Mutation):
    def __init__(self):
        super().__init__()
        self.scenario_generator_configuration: ScenarioGeneratorConfig
        self.scenario_generator_configuration = scenario_generation_utils.parse_scenario_generator_configuration_from_kwargs()
        # Container of enabled mutation operators for this generator
        self.planning_problem_mutation_operators = [scenario_mutation.mutate_initial_position,
                                                    scenario_mutation.mutate_initial_velocity,
                                                    scenario_mutation.mutate_goal_area_position]
        self.add_random_planning_problem = scenario_mutation.add_random_planning_problem
        self.add_random_planning_problem_prob = 0.3
        self.remove_random_planning_problem = scenario_mutation.remove_random_planning_problem
        self.remove_random_planning_problem_prob = 0.3
        self.scenario_validator_configuration = ValidationConfig()
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration)

    def _do(self, problem, X, **kwargs):
        print(f"\n{'=' * 75} START SCENARIO MUTATION {'=' * 75}")

        # Iterate over each individual (scenario, planning problem set)
        for i in range(len(X)):
            try:
                # With some probability add a planning problem.
                _scenario = deepcopy(X[i, 0])
                _planning_problem_set = deepcopy(X[i, 1])

                print(f"\n{'-' * 25} Mutation of scenario {_scenario.generation_id} {'-' * 25}\n")

                # TODO Does this have the is_valid flag?
                if hasattr(_scenario, "is_valid"):
                    delattr(_scenario, "is_valid")

                # Try to mutate the existing planning problems
                for key, planning_problem in _planning_problem_set.planning_problem_dict.items():
                    mutated_planning_problem = planning_problem

                    # Iterate over each planning problem mutation operator
                    for planning_problem_mutation_operator in self.planning_problem_mutation_operators:
                        # Apply the mutation operator with a probability of 1 / number of mutation operators
                        if random.random() <= 1 / len(self.planning_problem_mutation_operators):
                            print(f"Mutating Planning Problem {mutated_planning_problem.planning_problem_id} with {planning_problem_mutation_operator.__name__}")
                            if planning_problem_mutation_operator == scenario_mutation.mutate_evita_goal_area_position:
                                mutated_planning_problem = planning_problem_mutation_operator(self.scenario_generator_configuration,
                                                                                              mutated_planning_problem, _scenario, self.map_file_path)
                            else:
                                mutated_planning_problem = planning_problem_mutation_operator(self.scenario_generator_configuration,
                                                                                              mutated_planning_problem, _scenario)

                    # Save mutated planning problem
                    _planning_problem_set.planning_problem_dict[key] = mutated_planning_problem
                
                # Try to mutate the scenario's traffic lights if the scenario is urban
                if hasattr(self, "traffic_lights_mutation_operators"):
                    mutated_scenario = _scenario
                    # Iterate over each traffic light mutation operator
                    for traffic_light_mutation_operator in self.traffic_lights_mutation_operators:
                        if random.random() <= 1 / len(self.traffic_lights_mutation_operators):
                            # Apply the mutation operator with a probability of 1 / number of mutation operators
                            print(f"Mutating scenario {mutated_scenario.generation_id} with {traffic_light_mutation_operator.__name__}")
                            mutated_scenario = traffic_light_mutation_operator(self.scenario_generator_configuration, _scenario, self.map_file_path)

                    _scenario = mutated_scenario

                # Possibly add a new planning problem:
                # - If we haven't reached the maximum amount of planning problems yet
                # - With a certain probability to avoid always adding a planning problem at each mutation
                if len(_planning_problem_set.planning_problem_dict) < self.scenario_generator_configuration.max_amount_pp and \
                        random.random() <= self.add_random_planning_problem_prob:
                    print(f"Mutating {mutated_planning_problem.planning_problem_id} with {self.add_random_planning_problem.__name__}")
                    _planning_problem_set = self.add_random_planning_problem(self.scenario_generator_configuration, _scenario,
                                                                            _planning_problem_set, self.map_file_path)
                    
                # Possibly remove an existing planning problem - note there's the risk of removing the one we just added
                # - If we haven't reached the minimum amount of planning problems yet
                # - With a certain probability to avoid always removing a planning problem at each mutation
                if len(_planning_problem_set.planning_problem_dict) > self.scenario_generator_configuration.min_amount_pp and \
                        random.random() <= self.remove_random_planning_problem_prob:
                    print(f"Mutating {mutated_planning_problem.planning_problem_id} with {self.remove_random_planning_problem.__name__}")
                    _planning_problem_set = self.remove_random_planning_problem(self.scenario_generator_configuration,
                                                                                _scenario, _planning_problem_set)

                # Check mutated scenario validity
                self.scenario_validator.validate(_scenario, _planning_problem_set)

                if _scenario.is_valid:
                    X[i, 0] = _scenario
                    X[i, 1] = _planning_problem_set
                else:
                    print(f"Scenario {_scenario} is not valid!")
            
            except Exception as exc:
                print(f"Exception while mutating individual. {exc}")
                raise

        return X

class EVITAScenarioMutation(ScenarioMutation):
    def __init__(self, map_file_path: str):
        super().__init__()

        self.traffic_lights_mutation_operators = [scenario_mutation.mutate_tl_initial_configuration,
                                                  scenario_mutation.mutate_green_light_duration]
        
        self.planning_problem_mutation_operators.remove(scenario_mutation.mutate_goal_area_position)
        self.planning_problem_mutation_operators.append(scenario_mutation.mutate_evita_goal_area_position)
        
        
        self.add_random_planning_problem = scenario_mutation.add_evita_random_planning_problem
        
        self.map_file_path = map_file_path

        self.scenario_validator = EVITAScenarioValidator(self.scenario_validator_configuration)

class ScenarioCrossover(Crossover):
    def __init__(self):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.scenario_validator_configuration = ValidationConfig()
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration)

    def _do(self, problem, X, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output with the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):
            parent_1_scen, parent_2_scen, parent_1_planning_problem, parent_2_planning_problem = X[0, k, 0], X[1, k, 0], X[0, k, 1], X[1, k, 1]

            # Swap the scenarios objects
            offspring_1_scen = deepcopy(parent_2_scen)
            offspring_2_scen = deepcopy(parent_1_scen)

            # Get the min and max number of planning problems
            len_parent_1_planning_problem = len(parent_1_planning_problem.planning_problem_dict.values())
            len_parent_2_planning_problem = len(parent_2_planning_problem.planning_problem_dict.values())
            min_planning_problem = min(len_parent_1_planning_problem, len_parent_2_planning_problem)
            max_planning_problem = max(len_parent_1_planning_problem, len_parent_2_planning_problem)

            # Randomly choose a cut index between 0 and the minimum number of planning problems
            cut_index = random.randint(0, min_planning_problem - 1)
            
            offspring_1_planning_problem = list()
            offspring_2_planning_problem = list()
            # Swap the first `cut_index` planning problems between the two sets
            for i in range(cut_index):
                offspring_1_planning_problem.append(list(parent_2_planning_problem._planning_problem_dict.values()).__getitem__(i))
                offspring_2_planning_problem.append(list(parent_1_planning_problem._planning_problem_dict.values()).__getitem__(i))
            # For the remaining planning problems, keep the ones from the parent with more planning problems
            for i in range(cut_index, max_planning_problem):
                if len_parent_2_planning_problem > i:
                    offspring_1_planning_problem.append(list(parent_2_planning_problem._planning_problem_dict.values()).__getitem__(i))
                if len_parent_1_planning_problem > i:
                    offspring_2_planning_problem.append(list(parent_1_planning_problem._planning_problem_dict.values()).__getitem__(i))

            # Create the offspring planning problem sets
            planningProblemSet_offspring_1 = PlanningProblemSet(offspring_1_planning_problem)
            planningProblemSet_offspring_2 = PlanningProblemSet(offspring_2_planning_problem)

            # Check the validity of the offspring scenarios and planning problem sets.
            self.scenario_validator.validate_all([ (offspring_1_scen, planningProblemSet_offspring_1), (offspring_2_scen, planningProblemSet_offspring_2)])

            # If both offspring are valid, keep them. Otherwise, keep the parents.
            if offspring_1_scen.is_valid and offspring_2_scen.is_valid:
                # Return the children
                Y[0, k, 0] = offspring_1_scen
                Y[1, k, 0] = offspring_2_scen
                Y[0, k, 1] = planningProblemSet_offspring_1
                Y[1, k, 1] = planningProblemSet_offspring_2
            else:
                # Return the parents instead
                Y[0, k, 0] = parent_1_scen
                Y[1, k, 0] = parent_2_scen
                Y[0, k, 1] = parent_1_planning_problem
                Y[1, k, 1] = parent_2_planning_problem
                
        return Y

class EVITAScenarioCrossover(ScenarioCrossover):
    def __init__(self):
        super().__init__()
        self.scenario_validator = EVITAScenarioValidator(self.scenario_validator_configuration)

class ScenarioDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return a.X[0] == b.X[0] and a.X[1] == b.X[1]

# Generator to list all the simulation folders in this run
from ..utils import simulation_folders

from ..data_analysis import generate_report, generate_apollo_report
import os
import csv

def checkpoint(checkpoint_id: int, output_folder: str,
               all_scenarios:int, valid_scenarios:int, invalid_scenarios:int, total_generation_time:int, apollo: bool) -> None:
    """
    Link all the scenario folders generated so far + compute the results for the generation (aka csv files) to the checkpoint folder.
    """
    checkpoint_folder = os.path.join(output_folder, "checkpoints", f"{str(checkpoint_id).rjust(4,'0')}").replace("/records/", "/evita-records/")
    print(f"Creating checkpoint at {checkpoint_folder}")
    
    # Create a checkpoint folder
    os.makedirs(checkpoint_folder, exist_ok=False)

    # Create a scenarios.csv file containining the id of all the scenarios generated so far
    scenario_file = os.path.join(checkpoint_folder, "scenarios.csv")
    with open(scenario_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if apollo:
            csv_writer.writerow(["generation_id", "scenario_id", "scenario_file", "planned_trajectories_file"])
        else:
            csv_writer.writerow(["scenario_id", "scenario_file", "planned_trajectories_file"])
        for _, simulation_folder in simulation_folders(output_folder):
            if apollo:
                scenarios_folders = os.listdir(os.path.join(output_folder, simulation_folder))
                for scenario_id in scenarios_folders:
                    scenario_file = os.path.join(output_folder, simulation_folder, scenario_id, "scenario.xml").replace("/records/", "/evita-records/")
                    planned_trajectories_file = os.path.join(output_folder, simulation_folder, scenario_id, "planned_trajectories.json").replace("/records/", "/evita-records/")
                    csv_writer.writerow([str(simulation_folder), str(scenario_id), str(scenario_file), str(planned_trajectories_file)])
            else:
                scenario_id = simulation_folder
                simulation_db = os.path.join(output_folder, simulation_folder, "logs", "simulation.db")
                scenario_file = os.path.join(output_folder, simulation_folder, "scenario", f"scenario_{simulation_folder}.xml")
                csv_writer.writerow([str(scenario_id), str(scenario_file), str(simulation_db)])

    performance_file = os.path.join(checkpoint_folder, "performance.csv")
    with open(performance_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["all_scenarios", "valid_scenarios", "invalid_scenarios", "total_generation_time"])
        csv_writer.writerow([str(all_scenarios), str(valid_scenarios), str(invalid_scenarios), str(total_generation_time)])

    # Generate the usual report in the checkpoint folder using whatever data is available at this point
    if apollo:
        generate_apollo_report(output_folder, checkpoint_folder)
    else:
        generate_report(output_folder, checkpoint_folder)

class CustomTimeTermination(Termination):

    def __init__(self, max_time_minutes: int, checkpoint_every = 60, apollo: bool = False) -> None:
        super().__init__()

        # Do a checkpoint every 60 minutes
        self.checkpoint_every = checkpoint_every
        
        # Force this later! Circular dep, but cannot do otherwise
        self.problem_ref = None
        # Already completed checkpoints
        self.checkpoints = set()

        # This boolean value guides the type of files used to save checkpoints infos
        self.apollo = apollo

        self.max_time_seconds = max_time_minutes * 60
        self.elapsed_time_seconds = 0.0
        self.last_checkpoint = time.time()

    def update_elapsed_time(self):
        new_checkpoint = time.time()
        self.elapsed_time_seconds = self.elapsed_time_seconds + (new_checkpoint - self.last_checkpoint)
        self.last_checkpoint = new_checkpoint

    def restart(self):
        self.last_checkpoint = time.time()

    def _update(self, algorithm):

        checkpoint_id = int( self.elapsed_time_seconds / ( self.checkpoint_every * 60 ) )
        if not self.apollo and checkpoint_id > 0 and checkpoint_id not in self.checkpoints:
            # budget.pause()
            try:
                output_folder = self.problem_ref.output_folder
                all_scenarios = self.problem_ref.valid_scenarios
                valid_scenarios = self.problem_ref.valid_scenarios
                invalid_scenarios = self.problem_ref.invalid_scenarios
                self.update_elapsed_time()
                total_generation_time = self.elapsed_time_seconds

                checkpoint(checkpoint_id, output_folder,
                        all_scenarios, valid_scenarios, invalid_scenarios, total_generation_time, self.apollo)
                self.checkpoints.add(checkpoint_id)
            except Exception as e:
                # TODO Remove this
                print(f"Failed to checkpoint execution at checkpoint {checkpoint_id}")
                raise e

            # budget.resume()


        self.update_elapsed_time()
        return self.elapsed_time_seconds/self.max_time_seconds