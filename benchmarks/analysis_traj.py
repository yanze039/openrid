import matplotlib.pyplot as plt
import numpy as np


CV_data_file = "../ala2_out2/round_0/label/conf_0_0_26.CV.txt"
CV_data = np.loadtxt(CV_data_file)

running_average = np.zeros_like(CV_data)
for i in range(len(CV_data)):
    if i< 10:
        start_frame = i
    else:
        start_frame = int(i*0.1)
    running_average[i] = np.mean(CV_data[start_frame:i+1], axis=0)

plt.figure()
plt.plot(running_average[:,0], label="CV1")
plt.hlines(CV_data[0,0], 0, len(running_average), linestyles="dashed", color="orange", label="Initial CV")
plt.xlabel("steps")
plt.ylabel("Running average")
plt.legend()
plt.savefig("mean_force_converge_1.png")

plt.figure()
plt.plot(running_average[:,1], label="CV2")
plt.hlines(CV_data[0,1], 0, len(running_average), linestyles="dashed", color="orange", label="Initial CV")
plt.xlabel("steps")
plt.ylabel("Running average")
plt.legend()
plt.savefig("mean_force_converge_2.png")