import itertools
import math

import traci
import numpy as np
from IRS import IRS

class Environment:
    highway_length = 120


    no_vehicles = 4 #[4:7, 5:6, 6:5, 7:4]

    t = 0
    T = 30
    total_reward = 0
    total_served = 0
    total_requested_amount = 0
    list_of_vehicles = {}



    max_no_vehicles =4 #[4:6, 5:6, 6:5, 7:4]
    served_simultanously = 3
    action_sets = []
    no_actions = int(math.factorial(max_no_vehicles) / (math.factorial(served_simultanously) * math.factorial(max_no_vehicles - served_simultanously)))

    observation = []
    observation_length = 3 * max_no_vehicles + 1

    uav_total_reward = 0
    rsu_total_reward = 0
    requests = []
    vehicle_counter = 0

    delta_t = 1

    bitrates = []

    jain = 0

    no_uav = False
    excluded_vehicles = []

    vehicles_positions = None
    vehicles_speeds = None

    lefters = []

    SUMO_iteration = 100
    SUMO_iteration_counter = 0

    irs = None
    M = 50
    control_bit = 3

    def __init__(self):
        self.initalize()

    def initalize(self):
        self.action_sets = []
        for i in list(itertools.product([0, 1], repeat=self.max_no_vehicles)):
            if(sum(i) == self.served_simultanously):
                self.action_sets.append(i)
        print(self.action_sets)


        self.irs = IRS(self.served_simultanously, self.M, self.control_bit)


        self.vehicles_positions = [[[0 for x in range(self.T)] for y in range(self.no_vehicles)] for m in
                                   range(self.SUMO_iteration)]
        self.vehicles_speeds = [[[0 for x in range(self.T)] for y in range(self.no_vehicles)] for m in
                                range(self.SUMO_iteration)]

        self.initial_run()

    def reset(self):

        self.t = 0
        self.total_reward = 0
        self.total_served = 0
        self.lefters = []
        self.list_of_vehicles = {}

        self.vehicle_counter = 0
        self.bitrates = []
        self.jain = 0
        observation = np.zeros(self.observation_length)
        self.excluded_vehicles = []

        self.SUMO_iteration_counter += 1

        if (self.SUMO_iteration_counter == self.SUMO_iteration):
            self.SUMO_iteration_counter = 0

        return observation;

    def step(self, action):
        reward = 0
        observation_ = [-1] * self.observation_length
        # rsu_content = int(math.floor(action / (len(self.UAV_velocities))))

        IRS_vehicle_list = []

        action_set = self.action_sets[action]

        done = False

        counter=0
        not_selected = []
        for i in range(len(self.vehicles_positions[self.SUMO_iteration_counter])):
            if (self.vehicles_positions[self.SUMO_iteration_counter][i][self.t] == 0 or i in self.lefters):
                continue
            if (i not in self.list_of_vehicles):
                self.list_of_vehicles[i] = {"requested": 10000, "download": 0, "position": 0, "id": i, 'speed':0, "arrival": self.t}

            self.list_of_vehicles[i]["position"] = self.vehicles_positions[self.SUMO_iteration_counter][i][self.t]
            if (counter < self.max_no_vehicles):
                if(action_set[counter] == 1):
                    IRS_vehicle_list.append(self.list_of_vehicles[i])

                else:
                    not_selected.append(self.list_of_vehicles[i])
            counter+=1

        # while(len(IRS_vehicle_list) < self.served_simultanously and len(not_selected) > 0):
        #     IRS_vehicle_list.append(not_selected[0])
        #     not_selected.pop(0)


        if(len(IRS_vehicle_list) > 0):
            self.irs.serve(IRS_vehicle_list, optimize=True)



        self.t = self.t + self.delta_t

        observation_counter = 1

        no_vehicles_in_observation = 0
        for i in range(len(self.vehicles_positions[self.SUMO_iteration_counter])):
            if (self.vehicles_positions[self.SUMO_iteration_counter][i][self.t] == 0):
                continue

            if (self.vehicles_positions[self.SUMO_iteration_counter][i][self.t] > self.highway_length - 20
                    and i not in self.lefters):

                x = self.list_of_vehicles[i]['download'] / (self.t - self.list_of_vehicles[i]['arrival'])
                if(i == 0):
                    reward = x
                else:
                    if(self.total_reward > x):
                        reward = x - self.total_reward
                    else:
                        reward = 0

                self.bitrates.append(x)
                self.lefters.append(i)
                self.total_reward += reward
                self.total_reward = round(self.total_reward, 2)

            if (i not in self.list_of_vehicles):
                self.list_of_vehicles[i] = {"requested": 10000, "download": 0, "position": 0, "id": i, 'speed':0, "arrival": self.t}

            self.list_of_vehicles[i]["position"] = self.vehicles_positions[self.SUMO_iteration_counter][i][self.t]
            if (i not in self.lefters and no_vehicles_in_observation < self.max_no_vehicles):

                self.list_of_vehicles[i]["speed"] = self.vehicles_speeds[self.SUMO_iteration_counter][i][self.t]

                #Build Observation vector
                observation_[observation_counter] = int(self.list_of_vehicles[i]["position"])
                if(self.t > self.list_of_vehicles[i]['arrival']):
                    observation_[observation_counter + 1] = round(self.list_of_vehicles[i]["download"]/(self.t - self.list_of_vehicles[i]['arrival']), 2)
                observation_[observation_counter + 2] = int(self.list_of_vehicles[i]["speed"])

                observation_counter += 3

                no_vehicles_in_observation+=1

        observation_[0] = self.total_reward
        #print(self.t, observation_)

        if (self.t == self.T - 1):
            self.jain = np.sum(self.bitrates)**2/(len(self.bitrates) * np.sum(np.power(self.bitrates, 2)))
            done = True


        return reward, observation_, done


    def serveAndPhaseShift(self, action):
        action


    def initial_run(self):
        traci.start(["sumo", "-c", "C:\\Users\\umroot\\OneDrive\\Desktop\\implementations\\paper 6\\SUMO\\config.sumocfg"])
        # for test in range(100):
        #     traci.simulationStep()

        for x in range(self.SUMO_iteration):
            mapping = []
            self.excluded_vehicles = []
            for t in range(self.T):
                if traci.simulation.getMinExpectedNumber() > 0:
                    for veh_id in traci.simulation.getDepartedIDList():
                        traci.vehicle.subscribe(veh_id, [traci.constants.VAR_POSITION])

                    positions = traci.vehicle.getAllSubscriptionResults()

                    for key in positions:
                        veh = positions.get(key)
                        if (t == 0 or (len(mapping) == self.no_vehicles and key not in mapping)):
                            self.excluded_vehicles.append(key)

                        if (key in self.excluded_vehicles):
                            continue

                        if (key not in mapping):
                            mapping.append(key)

                        self.vehicles_positions[x][mapping.index(key)][t] = veh[66][0]
                        self.vehicles_speeds[x][mapping.index(key)][t] = traci.vehicle.getSpeed(key)
                traci.simulationStep()
        print(mapping)
        traci.close()

        # print(self.vehicles_positions)
        # print(self.vehicles_speeds)
