import numpy as np
import matplotlib.pyplot as plt

x = [100,200,500,1000,2000,5000,10000]
y1 = [0.85687, 0.85446,0.85565,0.87212,0.88547,0.89636,0.90664]
y2 = [0.84514,0.85155,0.86331,0.86731,0.87598,0.89655,0.90738]
y3 = [0.85267,0.84025,0.86177,0.86835,0.88073,0.89489,0.90496]
plt.plot(x,y1,marker = "o", color = 'red')
plt.plot(x,y2,marker = "o", color = 'green')
plt.plot(x,y3,marker = "o", color = 'blue')
plt.title("Algorithm Accurancy",fontsize=20)
plt.xlabel("Number of samples in learning stage",fontsize=15)
plt.ylabel("Accurancy in testing stage",fontsize=15)
plt.xticks(x,x)
plt.xticks(rotation=45)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
#plt.yticks(y1,y1)
#plt.yticks(y2,y2)
#plt.yticks(y3,y3)

plt.xlim(0, 10200)
plt.ylim(bottom=0.83, top = 0.91)
plt.grid()
plt.show() 