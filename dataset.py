import numpy.random as rnd
import numpy as np
import csv
import matplotlib.pyplot as plt

# thresholds
th_max_tandelta = 10^-2
th_max_temperature = 30
th_max_current = 150
th_middle_tandelta = 10^-3
th_middle_temperature = 24
th_middle_current = 100
th_min_tandelta = 10^-4
th_min_temperature = 18
th_min_current = 50

# ranges
max_val_tandelta = 10
max_val_temperature = 35
max_val_current = 200
min_val_tandelta = 10^-6
min_val_temperature = 0
min_val_current = 0
max_val_time = 60
min_val_time = 0

# utils
num_values = 5000
values_temp_output = np.zeros(num_values)
values_output = np.empty(num_values)

# bathtub curve plot (tentativo per inserire una dipendenza dal tempo della probabilità di guasto tramite "BATHTUB CURVE = CURVA A VASCA DA BAGNO")
# interpolata in maniera molto indicativa
xvals = np.linspace(0,max_val_time,max_val_time)
y = [1, 0.6, 0.37, 0.37, 0.6, 1]
x = [0, 8, 15, 45, 52, 60]
bathtub_curve = np.interp(x,x,y)
"""
plt.plot(x,bathtub_curve,linestyle='--', label = 'bathtub curve')
plt.legend()
plt.xlim(min_val_time, max_val_time)
plt.ylim(bottom=0, top = 1)
plt.show()
"""


# CSV file - tandelta tempearature current time output (output 1 = failure)
values_tandelta = np.around(rnd.uniform(min_val_tandelta, max_val_tandelta, num_values),3)
values_temperature = np.around(rnd.uniform(min_val_temperature, max_val_temperature, num_values),3)
values_current = np.around(rnd.uniform(min_val_current, max_val_current, num_values),3)
values_time = rnd.randint(min_val_time, max_val_time, num_values)


for i in range(num_values):
    # 1 parametro supera soglia max
    if (values_tandelta[i] > th_max_tandelta or values_temperature[i] > th_max_temperature or
    values_current[i] > th_max_current): 
       if (np.interp(values_time[i],x,y) > 0.5): #se la probabilità di guasto (corrispondente agli anni di vita del giunto) è > 0.5
            values_temp_output[i] = 1
    # 2 parametri superano soglia middle
    elif ((values_tandelta[i] > th_middle_tandelta and values_temperature[i] > th_middle_temperature) or 
    (values_tandelta[i] > th_middle_tandelta and values_current[i] > th_middle_current) or 
    (values_temperature[i] > th_middle_temperature and values_current[i] > th_middle_current)):
        if (np.interp(values_time[i],x,y) > 0.5): #se la probabilità di guasto (corrispondente agli anni di vita del giunto) è > 0.5
            values_temp_output[i] = 1
    # 3 parametri superano soglia min
    elif (values_tandelta[i] > th_min_tandelta and values_temperature[i] > th_min_temperature and values_current[i] > th_min_current):
        if (np.interp(values_time[i],x,y) > 0.5): #se la probabilità di guasto (corrispondente agli anni di vita del giunto) è > 0.5
            values_temp_output[i] = 1

print(values_tandelta[1], values_temperature[1], values_current[1], values_time[1], values_temp_output[1]) # valori temporanei

# elaborazione random del dato di output (per tenere conto di errori di misura, incertenzza etc)

# generazione vettore di 100 elementi con 10 zeri in posizioni random 
random_values = np.ones(100)
count = 0
while count < 10:
    temp_index = round(rnd.uniform(0,99, None))
    if (random_values[temp_index]) == 1:
        random_values[temp_index] = 0
        count = count + 1

#aggiornamento dati di output
# moltiplico per un valore (0 o 1) preso random dal vettore
for i in range(num_values):
    values_output[i] = values_temp_output[i]*random_values[round(rnd.uniform(0,99, None))] 

print(values_tandelta[1], values_temperature[1], values_current[1], values_time[1], values_output[1]) # valori finali

with open('C:/Users/Virginia/Desktop/MLAlgorithms/dataset.csv','w',newline='') as file:
    writer = csv.writer(file)
    for i in range(num_values):
        writer.writerow([values_tandelta[i], values_temperature[i], values_current[i], values_time[i], values_output[i]])
