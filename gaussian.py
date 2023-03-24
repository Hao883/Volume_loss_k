import numpy as np
import math
import os
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

parent_path = r"C:\Users\Hao.Shen\OneDrive - Tetra Tech, Inc\Documents\WSA\XDisp CP stage\curve fitting\inputs"
dir_list = os.listdir(parent_path)

# Define the function to fit
# Diameter (m)
D = 7.14
# Depth (m)


XP = pd.DataFrame(columns=['Model ID','Depth (m)', 'Volume Loss (%)', 'Trough width parameter k'])
XP['Model ID'] = dir_list
XP['Depth (m)'] = [17.8, 17.8,15.3,17,17, 31,31,18.1,18.1,23.9,23.9,27]


def Gau(x, VL, k):
    i = k * z0
    return VL * math.pi * D * D /(4 * i * np.sqrt(2*math.pi)) * np.exp(-x*x/(2*i*i))

# import data
for settlement in dir_list:
    path = r"C:\Users\Hao.Shen\OneDrive - Tetra Tech, Inc\Documents\WSA\XDisp CP stage\curve fitting\inputs\{}".format(settlement)
    filename_with_ext = os.path.basename(path)
    filename_without_ext = os.path.splitext(filename_with_ext)[0]
    df = pd.read_csv(path)
    x_data = df['distance'].to_numpy()
    y_data = np.abs(df['settlement'].to_numpy()*0.001)
    # Perform the regression
    z0 = XP.loc[XP['Model ID'] == settlement, "Depth (m)"].iloc[0]
    popt, pcov = curve_fit(Gau, x_data, y_data)
    print(popt)
    print('Volume loss:',popt[0]*100,'%')
    XP.loc[XP['Model ID'] == settlement, "Volume Loss (%)"] = popt[0]*100
    print('Trough width parameter:',popt[1])
    XP.loc[XP['Model ID'] == settlement, "Trough width parameter k"] = popt[1]


    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(x_data, y_data*1000, 'o', label='RS2 simluated settlement profile, provided by HATCH')
    ax.plot(x_data, Gau(x_data, *popt)*1000, 'g-', label='The Best Fit (the least squares method)')
    ax.set_xlabel('Distance transverse to Cross Passage (m)')
    ax.set_ylabel('Settlement (mm)')
    plt.title("Settlement profile {}".format(filename_without_ext))
    plt.legend()
    plt.legend(loc='lower center')
    plt.savefig("Best fit for settlement {}".format(filename_without_ext))
    # plt.show()
print(XP)
XP.to_csv("results.csv", index=False)
