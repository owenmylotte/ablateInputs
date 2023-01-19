import numpy as np  # for matrix manipulation
import matplotlib.pyplot as plt  # for plotting

kapparef = 1
Tsurf = 300.0

H2O_coeff = np.array([0.22317e1, -.15829e1, .1329601e1, -.50707, .93334e-1, -0.83108e-2, 0.28834e-3])
CO2_coeff = np.array([0.38041E1, -0.27808E1, 0.11672E1, -0.284910E0, 0.38163E-1, -0.26292E-2, 0.73662E-4])
CH4_coeff = np.array([6.6334E0, -3.5686E-3, 1.6682E-8, 2.5611E-10, -2.6558E-14])
CO_1_coeff = np.array([4.7869E0, -6.953E-2, 2.95775E-4, -4.25732E-7, 2.02894E-10])
CO_2_coeff = np.array([1.0091E1, -1.183E-2, 4.7753E-6, -5.87209E-10, -2.5334E-14])
MWC = 1.2010700e+01
MWO = 1.599940e+01
MWH = 1.007940e+00
UGC = 8314.4
MWCO = MWC + MWO
MWCO2 = MWC + 2. * MWO
MWCH4 = MWC + 4. * MWH
MWH2O = 2. * MWH + MWO

length = 350

# The Zimmer model uses a fit approximation of the absorptivity
kappaH2O = np.zeros(length)
kappaCO2 = np.zeros(length)
kappaCH4 = np.zeros(length)
kappaCO = np.zeros(length)
pCO2 = 0
pH2O = 0
pCH4 = 0
pCO = 0

temperature = np.linspace(0, 4000, length)
kappa = np.zeros(length)

for n in range(length):
    # Computing the Planck mean absorption coefficient for CO2 and H2O*/
    for j in range(7):
        kappaH2O[n] += H2O_coeff[j] * pow(temperature[n] / Tsurf, j)
        kappaCO2[n] += CO2_coeff[j] * pow(temperature[n] / Tsurf, j)

    kappaH2O[n] = kapparef * pow(10, kappaH2O[n])
    kappaCO2[n] = kapparef * pow(10, kappaCO2[n])

    # Computing the Planck mean absorption coefficient for CH4 and CO
    #  * The relationship is different with enough significance to use different models above or below 750 K.
    for j in range(5):
        kappaCH4[n] += CH4_coeff[j] * pow(temperature[n], j)
        if (temperature[n] <= 750):
            kappaCO[n] += CO_1_coeff[j] * pow(temperature[n], j)
        else:
            kappaCO[n] += CO_2_coeff[j] * pow(temperature[n], j)

        # Get the density mass fractions

    YinH2O = 0.1
    YinCO2 = 0.1
    YinCH4 = 0.1
    YinCO = 0.1

    density = 1

    # Computing the partial pressure of each species*/
    pCO2 = (density * UGC * YinCO2 * temperature) / (MWCO2 * 101325.)
    pH2O = (density * UGC * YinH2O * temperature) / (MWH2O * 101325.)
    pCH4 = (density * UGC * YinCH4 * temperature) / (MWCH4 * 101325.)
    pCO = (density * UGC * YinCO * temperature) / (MWCO * 101325.)

    # The resulting absorptivity is an average of species absorptivity weighted by partial pressure. */
    kappa[n] = 0
    if pCO2[n] > 1E-3 and kappaCO2[n] > 0:
        kappa[n] += pCO2[n] * kappaCO2[n]
    if pH2O[n] > 1E-3 and kappaH2O[n] > 0:
        kappa[n] += pH2O[n] * kappaH2O[n]
    if pCH4[n] > 1E-3 and kappaCH4[n] > 0:
        kappa[n] += pCH4[n] * kappaCH4[n]
    if pCO[n] > 1E-3 and kappaCO[n] > 0:
        kappa[n] += pCO[n] * kappaCO[n]

plt.figure(figsize=(6, 4), num=2)
plt.title("Zimmer Constants", pad=1, fontsize=10)  # TITLE HERE
# plt.plot(temperature, kappaH2O, c='blue', linewidth=1)
# plt.plot(temperature, kappaCO2, c='grey', linewidth=1)
# plt.plot(temperature, kappaCH4, c='red', linewidth=1)
# plt.plot(temperature, kappaCO, c='green', linewidth=1)

plt.semilogy(temperature, kappaH2O, c='blue', linewidth=1)
plt.semilogy(temperature, kappaCO2, c='grey', linewidth=1)
plt.semilogy(temperature, kappaCH4, c='red', linewidth=1)
plt.semilogy(temperature, kappaCO, c='green', linewidth=1)

plt.semilogy(temperature, kappa, c='black', linewidth=1)

plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel('Temperature [K]', fontsize=10)
plt.ylabel('Coefficient', fontsize=10)
plt.ylim(1E-1, 1E5)
# plt.xlim(0, 2500)
# plt.savefig('PlatesIrradiation_Convergence', dpi=1000, bbox_inches='tight')
plt.show()

print(min(kappaH2O))
print(temperature[np.argmin(kappaH2O)])
print(min(kappaCO2))
print(temperature[np.argmin(kappaCO2)])
print(min(kappaCH4))
print(temperature[np.argmin(kappaCH4)])
print(min(kappaCO))
print(temperature[np.argmin(kappaCO)])
