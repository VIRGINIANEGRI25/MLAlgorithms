import numpy.random as rnd
import numpy as np
import csv
import matplotlib.pyplot as plt

def main():
        
    # soglie singole, a coppie, di tutte e tre le grandezze 
    # da superare per avere guasto
    
    #tan delta
    th_1_tandelta = 10^-2
    th_2_tandelta = 10^-3
    th_3_tandelta = 10^-4
    # temperatura terreno
    th_1_temperature = 40 
    th_2_temperature = 30
    th_3_temperature = 25
    # consideriamo un cavo MT ch tipicamente ha portata 300A
    th_1_current = 300 
    th_2_current = 150
    th_3_current = 50

    # limiti di variazione
    # sup e inf per distribuzione uniforme 
    # a 3sigma per distribuzione normale
    max_val_tandelta = 10^-1
    min_val_tandelta = 10^-5
    max_val_temperature = 45 
    min_val_temperature = 5
    max_val_current = 300
    min_val_current = 0
    max_val_years = 30
    min_val_years = 0

    # utils

    # 1 2 5 numero di samples for learning
    # n_learn for other tests fixed at 1000
    n_learn = 10000

    # n_test always fixed at 100k
    n_test = 100000

    num_values = n_learn + n_test
    print("Numero campioni learning:", n_learn)
    print("Numero campioni test:", n_test)



    values_temp_output = np.zeros(num_values)
    values_output = np.zeros(num_values)

    # bathtub curve plot 
    # (tentativo per inserire una dipendenza dal tempo della probabilità di guasto 
    # tramite "BATHTUB CURVE = CURVA A VASCA DA BAGNO")
    # interpolata in maniera molto indicativa
    xvals = np.linspace(0,max_val_years,max_val_years)
    y = [1, 0.6, 0.37, 0.37, 0.6, 1]
    x = [0, 4, 8, 22, 26, 30]
    bathtub_curve = np.interp(x,x,y)

    # eventuale plot della curva 
    '''
    plt.plot(x,bathtub_curve,linestyle='--', label = 'Curva a vasca da bagno')
    plt.legend()
    plt.xlim(min_val_years, max_val_years)
    plt.ylim(bottom=0, top = 1)
    plt.xlabel("Età (anni)")
    plt.ylabel("Tasso di guasto")
    plt.show() 
    '''

    # CSV file - tandelta tempearature current years output (output 1 = failure)
    # uniform distribution
    # '''
    values_tandelta = np.around(rnd.uniform(min_val_tandelta, max_val_tandelta, num_values),3)
    values_temperature = np.around(rnd.uniform(min_val_temperature, max_val_temperature, num_values),3)
    values_current = np.around(rnd.uniform(min_val_current, max_val_current, num_values),3)
    # '''
    
    # CSV file - tandelta tempearature current years output (output 1 = failure)
    # normal distribution
    '''
    mu_tandelta = (min_val_tandelta+max_val_tandelta)/2
    mu_temperature = (min_val_temperature+max_val_temperature)/2
    mu_current = (min_val_current+max_val_current)/2
    sigma_tandelta = (max_val_tandelta-mu_tandelta)/3
    sigma_temperature = (max_val_temperature-mu_temperature)/3
    sigma_current = (max_val_current-mu_current)/3
    values_tandelta = np.around(rnd.normal(mu_tandelta, sigma_tandelta, num_values),3)
    values_temperature = np.around(rnd.normal(mu_temperature, sigma_temperature, num_values),3)
    values_current = np.around(rnd.normal(mu_current, sigma_current, num_values),3)
    '''

    values_years = rnd.randint(min_val_years, max_val_years, num_values)


    for i in range(num_values):
        # 1 parametro supera soglia 
        if (values_tandelta[i] > th_1_tandelta or values_temperature[i] > th_1_temperature or
        values_current[i] > th_1_current): 
            if (np.interp(values_years[i],x,y) > 0.5): #se la probabilità di guasto (corrispondente agli anni di vita del giunto) è > 0.5
                  values_temp_output[i] = 1
        # 2 parametri superano soglia 
        elif ((values_tandelta[i] > th_2_tandelta and values_temperature[i] > th_2_temperature) or 
        (values_tandelta[i] > th_2_tandelta and values_current[i] > th_2_current) or 
        (values_temperature[i] > th_2_temperature and values_current[i] > th_2_current)):
            if (np.interp(values_years[i],x,y) > 0.5): #se la probabilità di guasto (corrispondente agli anni di vita del giunto) è > 0.5
                values_temp_output[i] = 1
        # 3 parametri superano soglia 
        elif (values_tandelta[i] > th_3_tandelta and values_temperature[i] > th_3_temperature and values_current[i] > th_3_current):
            if (np.interp(values_years[i],x,y) > 0.5): #se la probabilità di guasto (corrispondente agli anni di vita del giunto) è > 0.5
                values_temp_output[i] = 1

    # stampa dei valori temporanei
    print(values_tandelta[1], values_temperature[1], values_current[1], values_years[1], values_temp_output[1])

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
    # aggiungo 10% di possibilità di errore
    for i in range(num_values):
        if (random_values[round(rnd.uniform(0,99, None))]==0):
            if (values_output[i]==0):
                values_output[i] = 1
            elif (values_output[i]==1):
                values_output[i] = 0

    # stampa valori finali
    print(values_tandelta[1], values_temperature[1], values_current[1], values_years[1], values_output[1]) 

    # conteggio di quanti sono 0 e quanti sono 1 nel learning dataset
    num_zeri = 0
    num_uni = 0
    for i in range(n_learn):
        if (values_output[i]==0):
            num_zeri=num_zeri+1
        elif (values_output[i]==1):
            num_uni=num_uni+1

    print("Numero di zeri nel dataset di learning:", num_zeri)
    print("Numero di uni nel dataset di learning:", num_uni)

    with open('C:/Users/Virginia/Desktop/MLAlgorithms/NostraBinaria/dataset.csv','w',newline='') as file:
        writer = csv.writer(file)
        for i in range(num_values):
            writer.writerow([values_tandelta[i], values_temperature[i], values_current[i], values_years[i], values_output[i]])

if __name__ == '__main__':
    main()