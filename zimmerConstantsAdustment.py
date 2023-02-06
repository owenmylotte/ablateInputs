import numpy as np  # for matrix manipulation
import matplotlib.pyplot as plt  # for plotting

# This is the code for the Zimmer absorption constants calculation

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
zKappaH2O = np.zeros(length)
zKappaCO2 = np.zeros(length)
zKappaCH4 = np.zeros(length)
zKappaCO = np.zeros(length)
# pCO2 = 0
# pH2O = 0
# pCH4 = 0
# pCO = 0

temperature = np.linspace(500, 2500, length)
# kappa = np.zeros(length)

for n in range(length):
    # Computing the Planck mean absorption coefficient for CO2 and H2O*/
    for j in range(7):
        zKappaH2O[n] += H2O_coeff[j] * pow(temperature[n] / Tsurf, j)
        zKappaCO2[n] += CO2_coeff[j] * pow(temperature[n] / Tsurf, j)

    zKappaH2O[n] = kapparef * pow(10, zKappaH2O[n])
    zKappaCO2[n] = kapparef * pow(10, zKappaCO2[n])

    # Computing the Planck mean absorption coefficient for CH4 and CO
    #  * The relationship is different with enough significance to use different models above or below 750 K.
    for j in range(5):
        zKappaCH4[n] += CH4_coeff[j] * pow(temperature[n], j)
        if (temperature[n] <= 750):
            zKappaCO[n] += CO_1_coeff[j] * pow(temperature[n], j)
        else:
            zKappaCO[n] += CO_2_coeff[j] * pow(temperature[n], j)

        # Get the density mass fractions

    YinH2O = 0.1
    YinCO2 = 0.1
    YinCH4 = 0.1
    YinCO = 0.1

    density = 1

    # # Computing the partial pressure of each species*/
    # pCO2 = (density * UGC * YinCO2 * temperature) / (MWCO2 * 101325.)
    # pH2O = (density * UGC * YinH2O * temperature) / (MWH2O * 101325.)
    # pCH4 = (density * UGC * YinCH4 * temperature) / (MWCH4 * 101325.)
    # pCO = (density * UGC * YinCO * temperature) / (MWCO * 101325.)

    # # The resulting absorptivity is an average of species absorptivity weighted by partial pressure. */
    # kappa[n] = 0
    # if pCO2[n] > 1E-3 and zKappaCO2[n] > 0:
    #     kappa[n] += pCO2[n] * zKappaCO2[n]
    # if pH2O[n] > 1E-3 and zKappaH2O[n] > 0:
    #     kappa[n] += pH2O[n] * zKappaH2O[n]
    # if pCH4[n] > 1E-3 and zKappaCH4[n] > 0:
    #     kappa[n] += pCH4[n] * zKappaCH4[n]
    # if pCO[n] > 1E-3 and zKappaCO[n] > 0:
    #     kappa[n] += pCO[n] * zKappaCO[n]

# Now get the emission of the gas based on an arbitrary path length
pL = 0.05  # Partial pressure times the path length through the gas. Assume 1 atm and 1 meter.
zEpsH2O = 1 - np.exp(-zKappaH2O * pL)
zEpsCO2 = 1 - np.exp(-zKappaCO2 * pL)
zEpsCH4 = 1 - np.exp(-zKappaCH4 * pL)
zEpsCO = 1 - np.exp(-zKappaCO * pL)

# This is the code for the sum of weighted gray gas constants model with MMA
wKappaH2O = np.zeros(length)
wKappaCO2 = np.zeros(length)
wKappaCH4 = np.zeros(length)
wKappaCO = np.zeros(length)

aH2O = max()

# Calculate the a[j] temperature factor

wEpsH2O = a[j](T) * (1 - np.exp(-wKappaH2O * pL))
wEpsCO2 = np.zeros(length)
wEpsCH4 = np.zeros(length)
wEpsCO = np.zeros(length)

wA = 0

# These are the plots

plt.figure(figsize=(6, 4), num=2)
plt.title("Absorption Constants", pad=1, fontsize=10)  # TITLE HERE
# plt.plot(temperature, kappaH2O, c='blue', linewidth=1)
# plt.plot(temperature, kappaCO2, c='grey', linewidth=1)
# plt.plot(temperature, kappaCH4, c='red', linewidth=1)
# plt.plot(temperature, kappaCO, c='green', linewidth=1)

plt.semilogy(temperature, zEpsH2O, c='blue', linewidth=1)
plt.semilogy(temperature, zEpsCO2, c='grey', linewidth=1)
plt.semilogy(temperature, zEpsCH4, c='red', linewidth=1)
plt.semilogy(temperature, zEpsCO, c='green', linewidth=1)

# plt.semilogy(temperature, kappa, c='black', linewidth=1)

plt.yticks(fontsize=7)
plt.xticks(fontsize=7)
plt.xlabel('Temperature [K]', fontsize=10)
plt.ylabel('Emissivity', fontsize=10)
plt.legend(["Z H2O", "Z CO2", "Z CH4", "Z CO"], loc="upper right", fontsize=7)
# plt.ylim(1E-1, 1E5)
# plt.xlim(0, 2500)
# plt.savefig('PlatesIrradiation_Convergence', dpi=1000, bbox_inches='tight')
plt.show()

print(min(zKappaH2O))
print(temperature[np.argmin(zKappaH2O)])
print(min(zKappaCO2))
print(temperature[np.argmin(zKappaCO2)])
print(min(zKappaCH4))
print(temperature[np.argmin(zKappaCH4)])
print(min(zKappaCO))
print(temperature[np.argmin(zKappaCO)])
