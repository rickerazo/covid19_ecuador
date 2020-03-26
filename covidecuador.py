####################################################################
# Ricardo Erazo
# Neuroscience Institute
# Georgia State University
# Emory University
# All rights reserved
####################################################################

############### 	COVID-19 in ECUADOR
# Code developed in python 3.6.8: objectives are:
#	1. to import excel spreadsheet of infected cases															-implemented
#	2. use an exponential function for a polynomial fit of data and predict infected cases in short term		-implemented
#	3. plot rate of infection 																					-implemented
#	4. use a logarithm regression to predict infected cases in short term 										-implemented
#	5. Account for suspected cases, not only officially published data: limited testing capacity 				-pending

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit
import datetime

p1 = pd.read_excel('covid19.xlsx', sheet_name='Sheet1', header=0, names=None, index_col=None, usecols=None, squeeze=False, dtype=None, engine=None, converters=None, true_values=None, false_values=None, skiprows=None, nrows=None, na_values=None, keep_default_na=True, verbose=False, parse_dates=False, date_parser=None, thousands=None, comment=None, skipfooter=0, convert_float=True, mangle_dupe_cols=True)

font = {'weight' : 'bold',
        'size'   : 40}
plt.rc('font', **font)

plt.close('all')


###################		DESCRIPTIVE STATS
plt.figure(figsize=(20,20))
plt.plot(p1.orden, p1.contagiados,'.-', linewidth = 8, label = 'contagiados', markersize = 40)
plt.plot(p1.orden, p1.muertos,':',linewidth = 5, label = 'muertos', color = 'k')
plt.plot(p1.orden, p1.recuperados, linewidth= 5, label = 'recuperados')


for i in np.arange(0,len(p1)-1,5):
	plt.text(p1.orden[i], -75, str(p1.meses[i])+ '/'+str(p1.dias[i]), fontsize = 30)
	# plt.text(p1.orden[0], -15, str(p1.meses[0])+ '/'+str(p1.dias[0]), fontsize = 20)
	# plt.text(p1.orden[5], -15, str(p1.meses[5])+ '/'+str(p1.dias[5]), fontsize = 20)
	# plt.text(p1.orden[10], -15, str(p1.meses[10])+ '/'+str(p1.dias[10]), fontsize = 20)

plt.text(p1.orden[len(p1)-1], -75, str(p1.meses[len(p1)-1])+ '/'+str(p1.dias[len(p1)-1]), fontsize = 30)


###################		INFERENTIAL STATS

## rate of infection
days = p1.orden.values 			#days since arrival or patient 1 to Ecuador
infected = p1.contagiados.values
rate_infection = np.diff(infected)

plt.plot(days[1:len(days)], rate_infection, label = 'tasa de infeccion', linewidth=8)

#######		Polinomial regression


def exponential_poly(days_since_patient1, contagiados):
	A = np.exp(days_since_patient1)* np.exp(days_since_patient1*contagiados)
	return A

# plt.plot

popt_e,pcov_e = curve_fit(exponential_poly, days, infected)

for i in range(1,3):
	days_future = np.append(days, days[-1]+i)
infected_predicted = exponential_poly(days_future, *popt_e)


plt.plot(days_future, infected_predicted, label = 'predicted', linewidth=8)

date_of_model_fit = datetime.datetime.now().date()
np.save(str(date_of_model_fit)+'_exponential_model.npy', infected_predicted)



plt.plot([16,16],[np.min(p1.contagiados), np.max(infected_predicted)],'--',linewidth = 10, label='estado de excepcion')
plt.plot([19,19],[np.min(p1.contagiados), np.max(infected_predicted)],'--',linewidth = 10, label='toque de queda aumentado')
plt.plot([26,26],[np.min(p1.contagiados), np.max(infected_predicted)],'--',linewidth = 10, label='toque de queda aumentado ++')


########### linear regression doesnt fit
# r = linregress(predictor1, muertos_predicted)
# plt.plot(predictor1, r.intercept + r.slope*predictor1, label = 'predicted trend')

plt.title('Covid-19 en Ecuador')
plt.legend(fontsize = 30, loc=2)

plt.xticks(np.arange(1,days_future[-1],5))

plt.ylabel('Gente', fontsize=50,fontweight='bold')
plt.xlabel('Dias desde paciente 1', fontsize=50,fontweight='bold')

# plt.ion()
# plt.show()

plt.savefig(str(date_of_model_fit)+'_covid19.png')




###########
# infected = 2 x / 3

## log scale graph:
fig2 = plt.figure(2, figsize=(20,20))
h2 = fig2.add_subplot(111)
h2.plot(days, np.log(infected),linewidth = 8, label='Infected')

############## Predictions:
#every two days
y = np.array([1]) 	#infected in day 1
x = np.array([1])
y1 = 1		#infected in day 1

time_for_infection_increase = 2
for i in np.arange(2,len(days)+time_for_infection_increase, time_for_infection_increase):
	y1 = y1*2
	y = np.append(y,y1)
	x = np.append(x,i)

h2.plot(x, np.log(y), linewidth = 5, label='2x every 2 days')


#every three days
y = np.array([1]) 	#infected in day 1
x = np.array([1])
y1 = 1		#infected in day 1

time_for_infection_increase = 3
for i in np.arange(2,len(days)+time_for_infection_increase+4, time_for_infection_increase):
	y1 = y1*2
	y = np.append(y,y1)
	x = np.append(x,i)

h2.plot(x, np.log(y), linewidth = 5, label='2x every 3 days')


# #every 2.5 days
# y = np.array([1]) 	#infected in day 1
# x = np.array([1])
# y1 = 1		#infected in day 1

# time_for_infection_increase = 2.5
# for i in np.arange(2,len(days)+time_for_infection_increase+4, time_for_infection_increase):
# 	y1 = y1*2
# 	y = np.append(y,y1)
# 	x = np.append(x,i)

# h2.plot(x, np.log(y), linewidth = 5, label='2x every 2.5 days')




########### text informative data
h2.text(7,1, 'Data: day '+str(days[-1])+', infected='+str(infected[-1]))

h2.text(7,0.8, 'Projected: day '+str(x[-1])+', infected='+str(y[-1]))

plt.legend(fontsize = 30, loc=2)

plt.yscale('log')

plt.savefig(str(date_of_model_fit)+'_covid19_logScale.png')

# plt.ion()
# plt.show()