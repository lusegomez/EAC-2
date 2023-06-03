import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats
import math
import statistics

data = np.loadtxt('fourierc.txt')
t = data[:, 0]
x = data[:, 1]

# plt.plot(x, t, 'o')
# plt.xlabel('x')
# plt.ylabel('t')
# plt.title('Gráfico de puntos (x, t)')
# plt.grid(True)
# plt.show()

# plt.hist(x, bins='auto', alpha=0.7, rwidth=0.85, label='Histograma')

# n, bins, patches = plt.hist(x, bins='auto', alpha=0)

# x_polygon = []
# y_polygon = []

# for i in range(len(bins) - 1):
#     x_polygon.append((bins[i] + bins[i+1]) /2)
#     y_polygon.append(n[i])

# plt.plot(x_polygon, y_polygon, 'r-', linewidth=2, label='Polígono de frecuencias')

# plt.legend()
# plt.xlabel('Valores de x')
# plt.ylabel('Frecuencia')
# plt.title('Histograma con polígono de frecuencias')

# plt.show()

def likelihood(a):
    #No agrego el termino sum(1/sqrt(2*pi*desvio**2)) ya que sera el mismo para todos los Xi
    mean = 0.1 * np.cos(2 * np.pi * a * Tn)
    residuals = Xn - mean
    likelihood = np.sum((Xn-mean) ** 2) #El denominador 2*desvio**2 es el mismo para todos los Xi
    return -likelihood


# Xn = 0.1 * np.cos(2 * np.pi * a * t) + Wn
Xn= x
Tn=t
a = 1.0 

initial_guess = 1.0
result = minimize(likelihood, initial_guess, method='BFGS')
estimated_a = result.x[0]
print("Estimador de máxima verosimilitud para a:", estimated_a)

#Ejercicio 2.1 ->
alpha = 5*10**5
Wn = np.random.randn(1000)
Xn_prima = 0.1 * np.cos(2 * np.pi * alpha * t) + Wn
err = 2.5*10**5
count = 0

def likelihood_2(a,i):
    mean = 0.1 * np.cos(2 * np.pi * a * Tn[i])
    residuals = Xn_prima[i] - mean
    likelihood = (residuals) ** 2
    return -likelihood

for i in range(len(Xn_prima)):
    result_2 = minimize(likelihood_2, err, (i), method='BFGS')
    if (result_2.x - err < 0):
        count += 1
    estimated_alpha = result_2.x

prob = count/len(Xn_prima)
print("Probabilidad de cometer un error:", prob)

#Ejercicio 2.2 ->
sum_Xn_prima = 0
mean = statistics.mean(Xn_prima)
diff = 0
for i in Xn_prima:
    diff += (i - mean) ** 2
sigma = math.sqrt(diff/(len(Xn_prima) - 1))

gamma =  (1 + 0.95) /2  

delta = (sigma/math.sqrt(len(Xn_prima))) * stats.norm.ppf(gamma, 0, 1)

print('[ %f , %f ]' %  (mean - delta, mean + delta))
