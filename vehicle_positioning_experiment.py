from IRS import IRS
import matplotlib.pyplot as plt
import numpy as np


while(True):

    irs = IRS(1, 100, 2)

    y_axis = []
    x_axis = []
    loc = 50
    vehicles = [{"position": loc, "download": 0,
                 "requested": 1000}]  # , {"position": 30, "download": 0, "requested": 1000}, {"position": 40, "download": 0, "requested": 1000}, {"position": 50, "download": 0, "requested": 1000}, {"position": 50, "download": 0, "requested": 1000}, {"position": 50, "download": 0, "requested": 1000}]

    # vehicles = [{"position": 20, "download": 0, "requested": 1000}]

    _, x = irs.serve(vehicles)


    first = 0
    last = 0
    its = 300
    c = 0

    for i in np.linspace(-150, 150, its):
        c = c + 1
        vehicles = [{"position":loc + (i * 0.01), "download":0, "requested":1000}]

        _, x = irs.serve(vehicles, optimize=False)

        y_axis.append(x) #i * 0.01 * 100
        x_axis.append(i * 0.01 * 100)


        if(c==1):
            first = x
        if(c==its):
            last = x

            if(abs(first - last) < first*0.002):
                plt.plot(x_axis, y_axis, linewidth=3, label='$x_v=$' + str(loc), color='g')

                plt.ylabel('Bit rate (bps/Hz)', fontsize=32)
                plt.xlabel('$\Delta$ (centimeters)', fontsize=32)
                plt.xticks(fontsize=32)
                plt.yticks(fontsize=32)
                plt.legend(prop={'size': 32})
                plt.show()