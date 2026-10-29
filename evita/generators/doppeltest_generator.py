import logging
import random
import enlighten
import traceback

from copy import deepcopy

from typing import Optional, List, Dict, Tuple

##### DoppelTest imports from main_ga.py
from copy import deepcopy
from random import random, sample, shuffle, randint

from DoppelTest.config import MAX_ADC_COUNT, MAX_PD_COUNT

from DoppelTest.framework.scenario import Scenario
from DoppelTest.framework.scenario.tc_config import TCSection
from DoppelTest.framework.scenario.pd_agents import PDAgent, PDSection
from DoppelTest.framework.scenario.ad_agents import ADAgent, ADSection
from DoppelTest.hdmap.MapParser import MapParser
from deap import base, tools, algorithms
import numpy as np

###

from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem, InitialState
from commonroad.scenario.lanelet import Lanelet

from ..interfaces import AbstractScenarioGenerator

from . import utils as scenario_generation_utils
from ..utils import EVITA_MAIN_LOGGER_NAME
from ..roads import identify_segments

from ..validator import ValidationConfig, ScenarioValidator

# MUTATION OPERATOR

def mut_ad_section(ind: ADSection):
    """
    Mutation operator for the ADSection of the scenario. It can do three types of mutation:
    - Remove a random AD agent (with a probability of 0.1)
    - Add a random AD agent (with a probability of 0.3)
    - Mutate a random AD agent (with a probability of 0.6)
    """
    mut_pb = random()

    # Remove a random AD agent with a probability of 10% and only if there are more than 2 AD agents (to avoid removing all AD agents)
    if mut_pb < 0.1 and len(ind.adcs) > 2:
        shuffle(ind.adcs)
        ind.adcs.pop()
        ind.adjust_time()
        return ind

    # Add a random AD agent with a probability of 40% and only if there are less than MAX_ADC_COUNT AD agents (to avoid adding too many AD agents)
    if mut_pb < 0.4 and len(ind.adcs) < MAX_ADC_COUNT:
        while True:
            new_ad = ADAgent.get_one()
            if ind.has_conflict(new_ad) and ind.add_agent(new_ad):
                break
        ind.adjust_time()
        return ind

    # Mutate a random AD agent
    index = randint(0, len(ind.adcs) - 1)
    routing = ind.adcs[index].routing
    original_adc = ind.adcs.pop(index)
    mut_counter = 0
    
    while True:
        if ind.add_agent(ADAgent.get_one_for_routing(routing)):
            break
        mut_counter += 1
        if mut_counter == 5:
            # mutation kept failing, dont mutate
            ind.add_agent(original_adc)
            pass

    ind.adjust_time()

    return ind

def mut_pd_section(ind: PDSection):
    """
    Mutation operator for the PDSection of the scenario. It can do three types of mutation:
    - Remove a random PD agent (with a probability of 0.2)
    - Add a random PD agent (with a probability of 0.2)
    - Mutate a random PD agent (with a probability of 0.6)
    """
    # If there are no crosswalk, do nothing
    ma = MapParser.get_instance()
    if len(list(ma.get_crosswalks())) == 0:
        return ind
    
    if len(ind.pds) == 0:
        ind.add_agent(PDAgent.get_one())
        return ind

    mut_pb = random()
    # remove a random
    if mut_pb < 0.2 and len(ind.pds) > 0:
        shuffle(ind.pds)
        ind.pds.pop()
        return ind

    # add a random
    if mut_pb < 0.4 and len(ind.pds) <= MAX_PD_COUNT:
        ind.pds.append(PDAgent.get_one())
        return ind

    # mutate a random
    index = randint(0, len(ind.pds) - 1)
    ind.pds[index] = PDAgent.get_one_for_cw(ind.pds[index].cw_id)

    return ind

def mut_tc_section(ind: TCSection):

    # If there are no Traffic Controllers, return ind    
    ma = MapParser.get_instance()
    if list(ma.get_signals()) == 0:
        return ind

    mut_pb = random()

    # Depending on the random probability, perform mutation on the initial configuration, final configuration, or duration of the traffic controller section
    if mut_pb < 0.3:
        ind.initial = TCSection.generate_config()
        return ind
    elif mut_pb < 0.6:
        ind.final = TCSection.generate_config()
    elif mut_pb < 0.9:
        ind.duration_g = TCSection.get_random_duration_g()

    return TCSection.get_one()

def mut_scenario(ind: Scenario):
    mut_pb = random()

    # Depending on the random probability, perform mutation on the ADSection, PDSection, or TCSection of the scenario
    if mut_pb < 1/3:
        ind.ad_section = mut_ad_section(ind.ad_section)
    elif mut_pb < 2/3:
        ind.pd_section = mut_pd_section(ind.pd_section)
    else:
        ind.tc_section = mut_tc_section(ind.tc_section)
    
    return ind

# CROSSOVER OPERATOR

def cx_ad_section(ind1: ADSection, ind2: ADSection):
    """
    Crossover operator for the ADSection of the scenario. It can do three types of crossover:
    - Swap the start_s or start_t of two AD agents with the same routing (with a probability of 0.05)
    - Add an AD agent from ind2 to ind1 if there exists a conflict (with a probability of 0.05)
    - Combine the AD agents from both parents to create a new ADSection (with a probability of 0.9)
    """
    
    cx_pb = random()

    if cx_pb < 0.05:
        return ind2, ind1

    cxed = False

    for adc1 in ind1.adcs:
        for adc2 in ind2.adcs:
            if adc1.routing_str == adc2.routing_str:
                # same routing in both parents
                # swap start_s and start_t
                if random() < 0.5:
                    adc1.start_s = adc2.start_s
                else:
                    adc1.start_t = adc2.start_t
                mutated = True
    if cxed:
        ind1.adjust_time()
        return ind1, ind2

    if len(ind1.adcs) < MAX_ADC_COUNT:
        for adc in ind2.adcs:
            if ind1.has_conflict(adc) and ind1.add_agent(deepcopy(adc)):
                # add an agent from parent 2 to parent 1 if there exists a conflict
                ind1.adjust_time()
                return ind1, ind2

    # if none of the above happened, no common adc, no conflict in either
    # combine to make a new populations
    available_adcs = ind1.adcs + ind2.adcs
    shuffle(available_adcs)
    split_index = randint(2, min(len(available_adcs), MAX_ADC_COUNT))

    result1 = ADSection([])
    for x in available_adcs[:split_index]:
        result1.add_agent(deepcopy(x))

    # make sure offspring adc count is valid

    while len(result1.adcs) > MAX_ADC_COUNT:
        result1.adcs.pop()

    while len(result1.adcs) < 2:
        new_ad = ADAgent.get_one()
        if result1.has_conflict(new_ad) and result1.add_agent(new_ad):
            break

    result1.adjust_time()
    
    return result1, ind2

def cx_pd_section(ind1: PDSection, ind2: PDSection):
    cx_pb = random()

    # Depending on the random probability, perform crossover on the PDSection of the scenarios
    if cx_pb < 0.1:
        return ind2, ind1

    available_pds = ind1.pds + ind2.pds

    result1 = PDSection(sample(available_pds, k=randint(0, min(MAX_PD_COUNT, len(available_pds)))))
    result2 = PDSection(sample(available_pds, k=randint(0, min(MAX_PD_COUNT, len(available_pds)))))
    
    return result1, result2

def cx_tc_section(ind1: TCSection, ind2: TCSection):
    cx_pb = random()

    # Depending on the random probability, perform crossover on the initial configuration, final configuration, or duration of the traffic controller section
    if cx_pb < 0.1:
        return ind2, ind1
    elif cx_pb < 0.4:
        ind1.initial, ind2.initial = ind2.initial, ind1.initial
    elif cx_pb < 0.7:
        ind1.final, ind2.final = ind2.final, ind1.final
    else:
        ind1.duration_g, ind2.duration_g = ind2.duration_g, ind1.duration_g

    return ind1, ind2

def cx_scenario(ind1: Scenario, ind2: Scenario):
    cx_pb = random()

    # Depending on the random probability, perform crossover on the ADSection, PDSection, or TCSection of the scenarios
    if cx_pb < 0.6:
        ind1.ad_section, ind2.ad_section = cx_ad_section(ind1.ad_section, ind2.ad_section)
    elif cx_pb < 0.6 + 0.2:
        ind1.pd_section, ind2.pd_section = cx_pd_section(ind1.pd_section, ind2.pd_section)
    else:
        ind1.tc_section, ind2.tc_section = cx_tc_section(ind1.tc_section, ind2.tc_section)

    return ind1, ind2

class DoppelTestScenarioGenerator(AbstractScenarioGenerator):
    
    # Maybe there's a way to convert the scenarios to MAPS so DoppelTest can use it.
    # Since we do not use APOLLO, Maybe it is better to tweak the MapParser and provide whatever info are needed for scenario generation
    # from commonroad instead
    # https://github.com/lgsvl/apollo-simulator-map/issues/1

    def __init__(self, commonroad_scenario_seed_file: Optional[str] = None, **kwargs) -> None:
        super(DoppelTestScenarioGenerator, self).__init__(commonroad_scenario_seed_file, **kwargs)
        
        assert commonroad_scenario_seed_file, "DoppelTestScenarioGenerator Generation requires a Seed File!"

        # Setup the logger (same as orchestrator.py)
        self.logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)        

        # Load the base scenario and planning problem set from the seed file
        self.base_scenario, self.base_planning_problem_set = scenario_generation_utils.get_base_scenario_from_file(commonroad_scenario_seed_file, remove_obstacles = True, remove_planning_problems = True)
        
        # Initialize the scenario generator configuration using the provided keyword arguments (debug settings, etc.)
        self.scenario_generator_configuration = scenario_generation_utils.parse_scenario_generator_configuration_from_kwargs(**kwargs)

        # Instantiate the scenario validator
        self.scenario_validator_configuration = ValidationConfig()
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration)

        self.generation = 0

        # DoppelTest specific setup - taken from main_ga.py

        # Parse the map
        MapParser(commonroad_scenario_seed_file)

        # GA Hyperparameters
        self.POP_SIZE = 10  # number of population
        self.OFF_SIZE = 10  # number of offspring to produce
        self.CXPB = 0.8  # crossover probablitiy
        self.MUTPB = 0.2  # mutation probability

        self.population = None # Current population
        self.offspring = None # Next generation
        self.hof = None

        self.toolbox = base.Toolbox()

        self.toolbox.register("mate", cx_scenario)
        self.toolbox.register("mutate", mut_scenario)
        self.toolbox.register("select", tools.selNSGA2)
    
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis = 0)
        self.stats.register("max", np.max, axis = 0)
        self.stats.register("min", np.min, axis = 0)
        self.logbook = tools.Logbook()
        self.logbook.header = 'gen', 'avg', 'max', 'min'

    def toCommonRoadFormat(self, individual) -> Tuple:
        
        # Initialize a new CommonRoad scenario by deep copying the base scenario, and an empty list to hold the planning problems for each agent in the individual scenario
        scenario = deepcopy(self.base_scenario)
        planning_problem_list = list()

        # Iterate over each agent in the ADSection
        for id, agent in enumerate(individual.ad_section.adcs):
            agent: ADAgent
            initial_lanelet: Lanelet
            initial_lanelet_id = agent.routing[0] # The first lanelet in the agent's routing is considered the initial lanelet for the agent's starting position
                
            try:
                # Find the initial lanelet in the CommonRoad scenario's lanelet network using the initial lanelet ID
                # Interpolate the agent's starting position along that lanelet based on the agent's start_s value
                initial_lanelet = scenario.lanelet_network.find_lanelet_by_id(initial_lanelet_id)
                initial_position = initial_lanelet.interpolate_position(agent.start_s)[0]
                
                # Determine the orientation of the agent based on its initial lanelet and position
                orientation = scenario_generation_utils.orientation_by_position(initial_lanelet, initial_position)

                # Create an InitialState object for the agent
                initial_state = InitialState(
                    time_step = int(agent.start_t / scenario.dt), # Convert from seconds to time steps
                    position = initial_position,
                    orientation = orientation,
                    velocity = 0.0,
                    acceleration = 0.0,
                    yaw_rate = 0.0,
                    slip_angle = 0.0
                )

                # Determine the goal lanelet and generate a goal region for the agent based on its routing
                goal_lanelet_id = agent.routing[-1]
                goal_lanelet = scenario.lanelet_network.find_lanelet_by_id(goal_lanelet_id)

                goal_area = scenario_generation_utils.get_possible_goal_for_lanelet(self.scenario_generator_configuration, goal_lanelet)
                goal_region = scenario_generation_utils.generate_goal(self.scenario_generator_configuration, goal_area)
                
                planning_problem = PlanningProblem(self.scenario_generator_configuration.av_base_id + id, initial_state, goal_region)
                
                planning_problem_list.append(planning_problem)
            except Exception as e:
                self.logger.critical(f"Cannot generate commonroad scenario!")
                self.logger.critical(traceback.format_exc())

                def debug_plot():
                    from commonroad.visualization.mp_renderer import MPRenderer
                    import matplotlib.pyplot as plt

                    rnd = MPRenderer()
                    scenario.lanelet_network.draw(rnd)
                    rnd.render()
                    
                    plt.plot(initial_position[0], initial_position[1], "o", color="green", zorder=100)
                    plt.plot(
                        [initial_position[0], initial_lanelet.center_vertices[-1][0], initial_lanelet.center_vertices[-2][0]],
                        [initial_position[1], initial_lanelet.center_vertices[-1][1], initial_lanelet.center_vertices[-2][1]],
                        "-", color="red", zorder=100)
                    
                    plt.plot(
                        [initial_position[0], initial_lanelet.center_vertices[0][0], initial_lanelet.center_vertices[1][0]],
                        [initial_position[1], initial_lanelet.center_vertices[0][1], initial_lanelet.center_vertices[1][1]],
                        "-", color="blue", zorder=100)

                    plt.show()

                # Use this only during MANUAL debugging
                # debug_plot()
                
                raise e
            
        # Convert the DoppelTest Scenario format into a (scenario, planning_problem_set) in CommonRoad format
        planning_problem_set = PlanningProblemSet(planning_problem_list)

        return (scenario, planning_problem_set)
        
    def generate(self, progress_bar_manager = None) -> List:
        """
        Return new scenarios (in CommonRoad) format
        """
        # Retrieve the progress bar manager from the argument or create a new one if not provided
        if progress_bar_manager:
            progress_bar_manager = progress_bar_manager
        else:
            progress_bar_manager = enlighten.get_manager()

        # Check if this is the first generation
        # Yes -> Generate the initial population
        # No -> Generate the offspring from the current population.
        if self.generation == 0:
            # Create a progress bar for the scenario generation process
            generation_bar = progress_bar_manager.counter(total = self.POP_SIZE, desc = "Scenario Generation", unit = "scenarios", color = "green", leave = False)

            # Generate the Initial Population
            # Scenario class requires a Parsed Map accessed statically via MapParser
            self.logger.info(f' ====== Generating the Initial Population ====== ')
            
            self.population = list()
            
            # Generate the initial population of scenarios using the DoppelTest framework
            for _ in range(self.POP_SIZE):
                self.population.append(Scenario.get_conflict_one())
                generation_bar.update()
            
            # Print newline to separate the progress bar from subsequent log messages
            print("\n")

            # Update the generation ID and chromosome ID for each scenario in the initial population
            for index, scenario in enumerate(self.population):
                scenario.gid = 0
                scenario.cid = index

            # Setup the Hall of Fame (HOF) to keep track of the best individuals found during the evolutionary process
            self.hof = tools.ParetoFront()

            # Close the progress bar after the initial population generation is complete
            generation_bar.close(clear = True)

            # Setup a way to match invalid_ind with the self.population
            self.individuals_to_evaluate = [ind for ind in self.population if not ind.fitness.valid]

            # Convert DoppelTest individuals to CommonRoad scenarios
            common_road_scenario_tuples = list()

            for individual in self.individuals_to_evaluate:
                try:
                    common_road_scenario_tuples.append(self.toCommonRoadFormat(individual))
                except Exception as e:
                    self.logger.critical("Cannot generate commonraod scenario, so force it to be invalid!")
                    
                    # Create a dummy CommonRoad scenario to represent the invalid individual, preserving the order of individuals in the population
                    faulty_scenario = deepcopy(self.base_scenario)
                    setattr(faulty_scenario, "is_valid", False)
                    setattr(faulty_scenario, "validation_message", "Cannot convert Apollo scenario to CommonRoad format!")
                    
                    common_road_scenario_tuples.append([faulty_scenario, None])

            return common_road_scenario_tuples
        else:
            self.logger.info(f' ====== Generating Generation {self.generation} ====== ')

            # Vary the population
            self.offspring = algorithms.varOr(self.population, self.toolbox, self.OFF_SIZE, self.CXPB, self.MUTPB)

            # Update chromosome gid and cid
            for index, scenario in enumerate(self.offspring):
                scenario.gid = self.generation
                scenario.cid = index

            # Evaluate the individuals with an invalid fitness
            self.individuals_to_evaluate = [individual for individual in self.offspring if not individual.fitness.valid]
            
            # Convert DoppelTest individuals to CommonRoad scenarios
            common_road_scenario_tuples = list()

            for individual in self.individuals_to_evaluate:
                try:
                    common_road_scenario_tuples.append(self.toCommonRoadFormat(individual))
                except Exception as e:
                    self.logger.critical("Cannot generate commonraod scenario, so force it to be invalid!")

                    # Create a dummy CommonRoad scenario to represent the invalid individual, preserving the order of individuals in the population
                    faulty_scenario = deepcopy(self.base_scenario)
                    setattr(faulty_scenario, "is_valid", False)
                    setattr(faulty_scenario, "validation_message", "cannot create a common road scenario!")

                    common_road_scenario_tuples.append((faulty_scenario, None))

            return common_road_scenario_tuples       

    def evaluate(self, list_of_simulation_data: List[Dict]) -> None:
        # According to the paper, DoppelTest requires four fitness functions:
        # - fmin distance: to encourage AV instances to be in close proximity with other road traffic participants to increase the likelihood of interaction;
        #     EVITA: We can track this
    
        # - fdecision: to maximize the number of decisions to be made by AV instances so that the scenario can be complex in a way that involves multiple AV instances making multiple decisions with regard to each other;
        #     EVITA: We cannot track this because the planner does not generate high-level action labels.

        # - fconflict: to maximize the number of conflicting trajectories between AV instances and other road traffic participants so that a collision violation may occur if an AV instance fails to yield right-of-way;
        #     EVITA: We can track this as we do in the scenario generation 

        # - fviolation: to maximize the number of violations that occur in a scenario
        #     EVITA: We can track collisions only. No traffic light, no stop line/sign, 

        # Compute the fitness values of the simulations and assign them to the list of individuals to evaluate
        for simulation_data, individual in zip(list_of_simulation_data, self.individuals_to_evaluate):
            # Extract data about agents and metrics
            scenario, planning_problem_set = simulation_data["individual"]
            agent_data = simulation_data["agent-data"]
            evaluation_data = simulation_data["evaluation-data"]

            # Default a very bad evaluation
            individual.fitness.values = (float('inf'), float('-inf'), float('-inf'), float('-inf'))

            # Compute the four metrics only if the scenario is valid        
            if scenario.is_valid:
                # Minimum of the minimum distance
                f_min_distance = evaluation_data["dce"].min()
                
                # Count how many vehicles collided during the simulation, not the number of accidents
                colliding_vehicles = set()
                for (agent_id, t) in evaluation_data.index:
                    if (evaluation_data.loc[(agent_id, t)]["has_collided"] == 1.0):
                        colliding_vehicles.add(agent_id)

                f_violations = len(colliding_vehicles)
                
                # This is the same as DoppelTest: report the amount of AV that have at least an intersection on their planned trajectories
                f_conflict = individual.has_ad_conflict()
                
                # Instead of decisions, we compute how many "segments" comprise the agent final trajectory.
                # The more segment the richer the movement
                # We argue that this is a low level metric to capture the same high-level concept of drivers's activity
                f_decisions = 0
                for agent in agent_data:
                    f_local_decisions, segments = identify_segments(agent)
                    f_decisions = f_decisions + f_local_decisions

                individual.fitness.values = (f_min_distance, f_decisions, f_conflict, f_violations)

                self.logger.critical(f"Scenario {scenario.generation_id} is valid")
            else:
                self.logger.critical(f"Scenario {scenario.generation_id} is not valid")

        # Depending whether we are in the first generation or not, the self.individuals_to_evaluate corresponds to the population or the offsprings
        if self.generation  == 0:
            self.logger.info(f' ====== Analyzing Initial Population ====== ')
            self.logger.debug(f"fitness values {[ind.fitness.values for ind in self.population]}")
            self.hof.update(self.population)
        else:
            self.logger.info(f' ====== Analyzing Generation {self.generation} ====== ')

            # Evaluate the offspring
            self.hof.update(self.offspring)

            self.logger.info(f' Population Size {len(self.population)}')
            self.logger.info(f' Offspring Size {len(self.offspring)}')
            self.logger.info(f' Target Size {self.POP_SIZE}')

            # Select the next generation population
            self.population[:] = self.toolbox.select(self.population + self.offspring, self.POP_SIZE)

            # Log the statistics of the current generation
            record = self.stats.compile(self.population)
            self.logbook.record(gen = self.generation, **record)
            
            # Log the statistics of the current generation to the logger
            self.logger.critical(self.logbook.stream)

        # Increment the generation count
        self.generation = self.generation + 1    
