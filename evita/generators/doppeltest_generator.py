"""
DoppelTest baseline (Huai et al., ICSE 2023) on CommonRoad maps.

DoppelTest's own genetic algorithm (NSGA-II from DEAP, with its scenario representation and operators) generates
scenarios; each is converted into a CommonRoad scenario (one planning problem per AD agent) and executed with FrenetiX.
The fitness is computed from the FrenetiX evaluation, as close as possible to DoppelTest's four objectives.
"""
import logging
import traceback
import enlighten
import numpy as np

from copy import deepcopy
from random import randint, random, sample, shuffle
from typing import Dict, List, Optional, Tuple
from deap import algorithms, base, tools

from commonroad.planning.planning_problem import InitialState, PlanningProblem, PlanningProblemSet
from commonroad.scenario.lanelet import Lanelet

from DoppelTest.config import MAX_ADC_COUNT, MAX_PD_COUNT
from DoppelTest.framework.scenario import Scenario
from DoppelTest.framework.scenario.ad_agents import ADAgent, ADSection
from DoppelTest.framework.scenario.pd_agents import PDAgent, PDSection
from DoppelTest.framework.scenario.tc_config import TCSection
from DoppelTest.hdmap.MapParser import MapParser

from . import utils as scenario_generation_utils
from ..interfaces import AbstractScenarioGenerator
from ..roads import identify_segments
from ..utils import EVITA_MAIN_LOGGER_NAME
from ..validator import ScenarioValidator, ValidationConfig

# ========== DoppelTest's genetic operators (from DoppelTest's main_ga.py, kept as published) ==========

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
    """
    Mutation operator for the TCSection (traffic lights): new initial configuration, final configuration or green
    duration. As published, the last two cases return a new random section instead of the mutated one.
    """
    # No traffic lights on the map: nothing to mutate (check added by EVITA)
    ma = MapParser.get_instance()
    if len(list(ma.get_signals())) == 0:
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
    """
    Mutate one of the three sections of a DoppelTest scenario, chosen with equal probability.
    """
    mut_pb = random()

    # Depending on the random probability, perform mutation on the ADSection, PDSection, or TCSection of the scenario
    if mut_pb < 1/3:
        ind.ad_section = mut_ad_section(ind.ad_section)
    elif mut_pb < 2/3:
        ind.pd_section = mut_pd_section(ind.pd_section)
    else:
        ind.tc_section = mut_tc_section(ind.tc_section)
    
    return ind, 

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
    """
    Crossover operator for the PDSection: swap the sections, or build two sections from random samples of both.
    """
    cx_pb = random()

    # Depending on the random probability, perform crossover on the PDSection of the scenarios
    if cx_pb < 0.1:
        return ind2, ind1

    available_pds = ind1.pds + ind2.pds

    result1 = PDSection(sample(available_pds, k=randint(0, min(MAX_PD_COUNT, len(available_pds)))))
    result2 = PDSection(sample(available_pds, k=randint(0, min(MAX_PD_COUNT, len(available_pds)))))
    
    return result1, result2

def cx_tc_section(ind1: TCSection, ind2: TCSection):
    """
    Crossover operator for the TCSection: swap the sections, or their initial or final configuration or green duration.
    """
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
    """
    Crossover of two DoppelTest scenarios on one of their sections (AD 60%, PD 20%, TC 20%).
    """
    cx_pb = random()

    # Depending on the random probability, perform crossover on the ADSection, PDSection, or TCSection of the scenarios
    if cx_pb < 0.6:
        ind1.ad_section, ind2.ad_section = cx_ad_section(ind1.ad_section, ind2.ad_section)
    elif cx_pb < 0.6 + 0.2:
        ind1.pd_section, ind2.pd_section = cx_pd_section(ind1.pd_section, ind2.pd_section)
    else:
        ind1.tc_section, ind2.tc_section = cx_tc_section(ind1.tc_section, ind2.tc_section)

    return ind1, ind2

# ========== Generator ==========

class DoppelTestScenarioGenerator(AbstractScenarioGenerator):
    """
    DoppelTest's genetic algorithm (population POP_SIZE, OFF_SIZE offspring per generation, crossover probability CXPB, mutation probability MUTPB, NSGA-II selection) on a CommonRoad map.
    generate() returns the individuals to evaluate (the initial population, then each generation's offspring) as CommonRoad scenarios;
    evaluate() receives their results in the same order.
    """

    def __init__(self, commonroad_scenario_seed_file: Optional[str] = None, **kwargs) -> None:
        super(DoppelTestScenarioGenerator, self).__init__(commonroad_scenario_seed_file, **kwargs)
        
        if not commonroad_scenario_seed_file:
            raise ValueError("DoppelTestScenarioGenerator Generation requires a Seed File!")

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
        self.individuals_to_evaluate = []

        self.toolbox = base.Toolbox()

        self.toolbox.register("mate", cx_scenario)
        self.toolbox.register("mutate", mut_scenario)
        self.toolbox.register("select", tools.selNSGA2)
    
        self.stats = tools.Statistics(key = lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis = 0)
        self.stats.register("max", np.max, axis = 0)
        self.stats.register("min", np.min, axis = 0)
        self.logbook = tools.Logbook()
        self.logbook.header = 'gen', 'avg', 'max', 'min'

    def toCommonRoadFormat(self, individual) -> Tuple:
        """
        CommonRoad (scenario, planning problem set) of a DoppelTest individual: one planning problem per AD agent,
        starting at rest at distance start_s along the first lanelet of its routing, at time start_t, with the goal
        area at the end of the last lanelet of its routing.
        """
        config = self.scenario_generator_configuration

        # Initialize a new CommonRoad scenario by deep copying the base scenario, and an empty list to hold the planning problems for each agent in the individual scenario
        scenario = deepcopy(self.base_scenario)
        planning_problem_list = list()

        # Iterate over each agent in the ADSection
        for id, agent in enumerate(individual.ad_section.adcs):
            agent: ADAgent
                
            try:
                # Find the initial lanelet in the CommonRoad scenario's lanelet network using the initial lanelet ID
                # Interpolate the agent's starting position along that lanelet based on the agent's start_s value
                initial_lanelet = scenario.lanelet_network.find_lanelet_by_id(agent.routing[0])
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
                    slip_angle = 0.0)

                # Determine the goal lanelet and generate a goal region for the agent based on its routing
                goal_lanelet = scenario.lanelet_network.find_lanelet_by_id(agent.routing[-1])
                goal_area = scenario_generation_utils.get_possible_goal_for_lanelet(config, goal_lanelet)
                goal_region = scenario_generation_utils.generate_goal(config, goal_area)
                
                planning_problem = PlanningProblem(config.av_base_id + id, initial_state, goal_region)
                planning_problem_list.append(planning_problem)
            except Exception as e:
                self.logger.critical(f"Cannot generate commonroad scenario!")
                self.logger.critical(traceback.format_exc())                
                raise
            
        # Convert the DoppelTest Scenario format into a (scenario, planning_problem_set) in CommonRoad format
        planning_problem_set = PlanningProblemSet(planning_problem_list)

        return scenario, planning_problem_set

    def _to_scenario_tuples(self, individuals: List) -> List[Tuple]:
        """
        CommonRoad scenarios of the individuals, in the same order.
        An individual that cannot be converted gives an invalid placeholder scenario (without planning problems), so the order is preserved.
        """
        scenario_tuples = []

        for individual in individuals:
            try:
                scenario_tuples.append(self.toCommonRoadFormat(individual))
            except Exception:
                self.logger.critical("Cannot generate commonroad scenario, so force it to be invalid!")
                faulty_scenario = deepcopy(self.base_scenario)
                setattr(faulty_scenario, "is_valid", False)
                setattr(faulty_scenario, "validation_message", "Cannot convert DoppelTest scenario to CommonRoad format!")
                scenario_tuples.append((faulty_scenario, None))

        return scenario_tuples
        
    def generate(self, progress_bar_manager = None, **kwargs) -> List:
        """
        The individuals to evaluate: the initial population at the first call, then the offspring of the current population, as CommonRoad scenarios.
        """
        # Check if this is the first generation
        # Yes -> Generate the initial population
        # No -> Generate the offspring from the current population.
        if self.generation == 0:
            # Create a progress bar for the scenario generation process
            progress_bar_manager = progress_bar_manager or enlighten.get_manager()
            generation_bar = progress_bar_manager.counter(total = self.POP_SIZE, desc = "Scenario Generation", unit = "scenarios", color = "green", leave = False)

            try:
                # Generate the Initial Population
                # Scenario class requires a Parsed Map accessed statically via MapParser
                self.logger.info(f' ====== Generating the Initial Population ====== ')
                
                self.population = []
                
                # Generate the initial population of scenarios using the DoppelTest framework
                for _ in range(self.POP_SIZE):
                    self.population.append(Scenario.get_conflict_one())
                    generation_bar.update()
            finally:
                generation_bar.close(clear = True)

            # Update the generation ID and chromosome ID for each scenario in the initial population
            for index, scenario in enumerate(self.population):
                scenario.gid = 0
                scenario.cid = index

            # Setup the Hall of Fame (HOF) to keep track of the best individuals found during the evolutionary process
            self.hof = tools.ParetoFront()
            candidates = self.population
        else:
            self.logger.info(f' ====== Generating Generation {self.generation} ====== ')
            self.offspring = algorithms.varOr(self.population, self.toolbox, self.OFF_SIZE, self.CXPB, self.MUTPB)

            for index, individual in enumerate(self.offspring):
                individual.gid = self.generation
                individual.cid = index
            
            candidates = self.offspring

        # Only individuals without a valid fitness need evaluating
        self.individuals_to_evaluate = [individual for individual in candidates if not individual.fitness.valid]

        return self._to_scenario_tuples(self.individuals_to_evaluate)   

    def evaluate(self, list_of_simulation_data: List[Dict]) -> None:
        """
        Assign the fitness of the evaluated individuals (results in the order of generate()), update the hall of fame, and select the next population.

        DoppelTest's four objectives, computed from the FrenetiX evaluation:
          - f_min_distance (minimized): smallest distance to collision (dce) reached by any vehicle;
          - f_decisions: number of turns of all vehicles' driven trajectories (DoppelTest counts the planner's
            decisions; FrenetiX has no high-level decision labels, so turns approximate them);
          - f_conflict: number of AD agents whose routes conflict (as in DoppelTest);
          - f_violations: number of vehicles that collided (FrenetiX has no traffic-rule oracles).
        Invalid scenarios get the worst values.
        """
        for simulation_data, individual in zip(list_of_simulation_data, self.individuals_to_evaluate):
            # Extract data about agents and metrics
            scenario, _ = simulation_data["individual"]
            evaluation_data = simulation_data["evaluation-data"]

            # Default a very bad evaluation
            individual.fitness.values = (float('inf'), float('-inf'), float('-inf'), float('-inf'))

            if not getattr(scenario, "is_valid", False) or evaluation_data is None:
                self.logger.critical(f"Scenario {getattr(scenario, 'generation_id', '?')} is not valid")
                continue

            # Compute the four metrics only if the scenario is valid  
                  
            # Minimum of the minimum distance
            f_min_distance = evaluation_data["dce"].min()
            if f_min_distance is None or np.isnan(f_min_distance):
                f_min_distance = float('inf')
                self.logger.debug(f"Scenario {scenario.generation_id} has no minimum distance, so set to infinity")

            # Count how many vehicles collided during the simulation, not the number of accidents
            collided = evaluation_data[evaluation_data["has_collided"] == 1.0]
            f_violations = len(set(collided.index.get_level_values(0)))
            
            # This is the same as DoppelTest: report the amount of AV that have at least an intersection on their planned trajectories
            f_conflict = individual.has_ad_conflict()
            
            # Instead of decisions, we compute how many "segments" comprise the agent final trajectory.
            # The more segment the richer the movement
            # We argue that this is a low level metric to capture the same high-level concept of drivers's activity
            f_decisions = sum(identify_segments(agent)[0] for agent in simulation_data["agent-data"])

            individual.fitness.values = (f_min_distance, f_decisions, f_conflict, f_violations)

            self.logger.critical(f"Scenario {scenario.generation_id} is valid")

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
        self.logger.critical(self.logbook.stream)

        self.generation += 1
