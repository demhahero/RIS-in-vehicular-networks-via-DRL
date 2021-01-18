import math
import cmath
import numpy as np
import random

class IRS:
    elements_phase_shift = None

    # RIS coordination
    I_x, I_y, I_z = 10, 20, 10

    # RSU  coordination
    R_x, R_y, R_z = 0, 40, 10


    ro = 10**-2
    # print(datarate)
    cascaded_gain = 0
    lamb = 1
    d = 0.5
    PU = 0.1
    sigma = 10 ** (-7)
    beta1 = 4

    distance_I_R = 0
    angle_I_R = 0

    phases_I_i = 0
    phase_R = 0

    distances_I_i = []
    angles_I_i = []

    M= 0

    control_bit = 3
    possible_angles = []

    served_simultanously = 2

    def __init__(self, served_simultanously, M, control_bit):
        self.M = M
        self.control_bit = control_bit
        self.possible_angles = np.linspace(0, 2 * math.pi, 2 ** self.control_bit, endpoint=False)

        self.elements_phase_shift_real = [random.choice(self.possible_angles) for x1 in range(self.M)]#np.random.uniform(low=-math.pi, high=math.pi, size=(self.M,))


        self.served_simultanously = served_simultanously

        self.elements_phase_shift_complex = np.zeros(self.M, dtype=np.complex_)
        self.phases_I_i = np.zeros([self.served_simultanously, self.M], dtype=np.complex_)
        self.phase_R = np.zeros(self.M, dtype=np.complex_)



        for m in range(self.M):
            self.elements_phase_shift_complex[m] = cmath.exp(self.elements_phase_shift_real[m] * 1j)
            self.phase_R[m] = cmath.exp(2 * (math.pi / self.lamb) * self.d * self.angle_I_R * (m) * 1j)

        self.distance_I_R = math.sqrt((self.R_x - self.I_x) ** 2 + (self.R_y - self.I_y) ** 2 + (self.R_z - self.I_z) ** 2)
        self.angle_I_R = (self.I_x - self.R_x) / self.distance_I_R




    def serve(self, vehicles, optimize=True):
        reward = 0

        self.compute_parms(vehicles)
        if(optimize == True):
            self.optimize_phase_shift(vehicles)

        for vehicle in range(len(vehicles)):
            if(vehicles[vehicle]['download'] < vehicles[vehicle]['requested']):
                vehicles[vehicle]['download'] = float(vehicles[vehicle]['download']) + self.compute_data_rate(vehicle)

                if(vehicles[vehicle]['download'] >= vehicles[vehicle]['requested']):
                    vehicles[vehicle]['download'] = vehicles[vehicle]['requested']
                    reward += vehicles[vehicle]['requested']
            # img = 0
            # for vehicle in range(len(vehicles)):
            #     for m in range(len(self.elements_phase_shift_complex)):
            #         img += self.elements_phase_shift_complex[m] * self.phases_I_i[vehicle][m] * self.phase_R[m]
        return reward , self.compute_data_rate(0)


    def compute_parms(self, vehicles):
        #Calculate vehicle to IRS distance and angles

        self.distances_I_i = [0] * len(vehicles)
        self.angles_I_i = [0] * len(vehicles)

        for vehicle in range(len(vehicles)):
            d_I_i = math.sqrt(
                (vehicles[vehicle]['position'] - self.I_x) ** 2 + (self.R_y - self.I_y) ** 2 + (1 - self.I_z) ** 2)
            self.distances_I_i[vehicle] = (d_I_i)
            self.angles_I_i[vehicle] = ((vehicles[vehicle]['position'] - self.I_x) / d_I_i)

        #Calculate phase shift with vehicles
        for m in range(len(self.elements_phase_shift_real)):
            for vehicle in range(len(vehicles)):
                self.phases_I_i[vehicle][m] = cmath.exp(-2 * (math.pi / self.lamb) * self.d * self.angles_I_i[vehicle] * (m) * 1j)

        #print(self.phases_I_i[vehicle])
        #print("D R:", self.distances_I_i)


    def optimize_phase_shift(self, vehicles):
        arr = []
        arr.append(self.optimize_compute_objective_function(vehicles))
        #BCD to optimize phase shifts

        for it in range(1):
            for m in range(self.M):
                best = 0
                best_phase = 0
                for phase in self.possible_angles:
                    self.elements_phase_shift_complex[m] = cmath.exp(phase * 1j)

                    x = self.optimize_compute_objective_function(vehicles)
                    if (best < x):
                        best = x
                        best_phase = cmath.exp(phase * 1j)

                self.elements_phase_shift_complex[m] = best_phase

            arr.append(best)


        #print(arr)



    def optimize_optimal(self, vehicles):
        arr = []
        arr.append(self.optimize_compute_objective_function(vehicles))
        #BCD to optimize phase shifts
        best = 0
        best_phase = 0
        for it in range(1):
            for m in range(len(self.elements_phase_shift_complex)):
                best = 0
                best_phase = 0
                for phase in np.linspace(0, 2 * math.pi, 2**3, endpoint=False):
                    self.elements_phase_shift_complex[m] = cmath.exp(phase * 1j)

                    if (best < self.optimize_compute_objective_function(vehicles)):
                        best = self.optimize_compute_objective_function(vehicles)
                        best_phase = cmath.exp(phase * 1j)

                self.elements_phase_shift_complex[m] = best_phase
            arr.append(best)
        #print(arr)


    def optimize_compute_objective_function(self, vehicles):
        sum_rate = 0
        for vehicle in range(len(vehicles)):
            img = 0
            #for m in range(self.M):
            #    img += self.elements_phase_shift_complex[m] * self.phases_I_i[vehicle][m] * self.phase_R[m]
            img = np.sum(np.multiply(np.multiply(self.elements_phase_shift_complex, self.phases_I_i[vehicle]), self.phase_R))

            cascaded_gain = (self.ro * img) / (math.sqrt(self.distances_I_i[vehicle] ** self.beta1) * math.sqrt(self.distance_I_R ** self.beta1))

            sum_rate += math.log(
                1 + self.PU * (np.abs(cascaded_gain) ** 2) / self.sigma ** 2)
        return sum_rate


    def compute_data_rate(self, vehicle):
        img = 0  # np.dot(np.dot(items, I_i), I_R)
        sum_rate = 0
        for m in range(self.M):
            comp = self.elements_phase_shift_complex[m] * self.phases_I_i[vehicle][m] * self.phase_R[m]
            img += comp


        cascaded_gain = (self.ro * img) / (math.sqrt(self.distances_I_i[vehicle] ** self.beta1) * math.sqrt(self.distance_I_R ** self.beta1))

        sum_rate += math.log(
            1 + self.PU * (np.abs(cascaded_gain) ** 2) / self.sigma ** 2)
        return sum_rate
