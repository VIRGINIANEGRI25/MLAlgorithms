from matplotlib.cbook import simple_linear_interpolation
import numpy as np
import matplotlib.pyplot as plt
import MyLR
import MyKnn
import MySvm
import MyAnn
import MyDecisionTree
import MyRandomForest

import dataset 
import csv


simulation_num = int(input('number of executions:'))


accurancies = np.empty(simulation_num)
count = 0
while count < simulation_num:
    print("iteration number:", count)
    dataset.main()
    accurancies[count] = MyLR.main()
    # accurancies[count] = MyKnn.main()
    # accurancies[count] = MySvm.main()
    # accurancies[count] = MyAnn.main()
    # accurancies[count] = MyDecisionTree.main()
    # accurancies[count] = MyRandomForest.main()

    count = count + 1

with open('C:/Users/Virginia/Desktop/MLAlgorithms/NostraBinaria/Simulazioni/Ripetibilità/LRAccurancy.csv','w',newline='') as file:
#with open('C:/Users/Virginia/Desktop/MLAlgorithms/NostraBinaria/Simulazioni/Ripetibilità/KnnAccurancy.csv','w',newline='') as file:
#with open('C:/Users/Virginia/Desktop/MLAlgorithms/NostraBinaria/Simulazioni/Ripetibilità/SvmAccurancy.csv','w',newline='') as file:
#with open('C:/Users/Virginia/Desktop/MLAlgorithms/NostraBinaria/Simulazioni/Ripetibilità/AnnAccurancy.csv','w',newline='') as file:
#with open('C:/Users/Virginia/Desktop/MLAlgorithms/NostraBinaria/Simulazioni/Ripetibilità/DecisionTreeAccurancy.csv','w',newline='') as file:
#with open('C:/Users/Virginia/Desktop/MLAlgorithms/NostraBinaria/Simulazioni/Ripetibilità/RandomForestAccurancy.csv','w',newline='') as file:

    writer = csv.writer(file)
    writer.writerows(map(lambda x: [x], accurancies))



executions =  np.linspace(0,simulation_num, simulation_num)
plt.plot(executions,accurancies, color = 'red')
plt.title("Algorithm Accurancy with 1000 samples in learning stage",fontsize=20)
plt.xlabel("Execution",fontsize=15)
plt.ylabel("Accurancy in testing stage",fontsize=15)
# plt.xticks(executions,executions)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
#plt.yticks(y,y)

plt.xlim(0, simulation_num)
#plt.ylim(bottom=0.850, top = 0.890)
#plt.ylim(bottom=0.866660, top = 0.87820)
plt.grid()
plt.show() 