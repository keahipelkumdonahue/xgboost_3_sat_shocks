import pandas as pd
import numpy as np
from visualizers import plot_hemisphere_heatmap
import matplotlib.pyplot as plt
from scipy.stats import norm

shocks = pd.read_csv('shocks_20250726_044117.dat', delimiter=',')  # list of helsinki shocks
shocks = shocks[['Shock type', 'Shock normal X', 'Shock normal Y', 'Shock normal Z', 'Shock speed (km/s)']]

earthward_vs = []
sunward_vs = []

for i in range(len(shocks)):
    shock = shocks.iloc[i]

    n_hat = np.array([shock['Shock normal X'], shock['Shock normal Y'], shock['Shock normal Z']])

    speed = np.abs(shock['Shock speed (km/s)'])

    v = speed * n_hat

    if v[0] > 0:
        sunward_vs.append(v)
    else:
        earthward_vs.append(v)

earthward_vs = np.array(earthward_vs)
sunward_vs = np.array(sunward_vs)

vs = np.array(sunward_vs)
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
plot_hemisphere_heatmap(vectors=vs, ax=ax, title='V-TRUE')
plt.show()

##########################################################################################
############## This code creates the training velocity component plot ####################
##########################################################################################

plt.style.use('dark_background')

fig = plt.figure(figsize=(8, 5))

plt.suptitle('Shock Velocity Component Distributions', fontweight='bold')

ax1 = plt.subplot(2, 3, 1)
data = earthward_vs[:, 0]
loc, sigma = norm.fit(data)
print(loc, sigma)
# plot data
plt.hist(data, density=True, bins=50, alpha=0.6, label='Observed Shock Components', color='#8059e7ff')
# plot fit
x = np.linspace(min(data), max(data), 500)
y = norm.pdf(x, loc, sigma)
plt.plot(x, y, label='Gaussian Fit', c='red')
plt.xlabel('Earthward GSE Vx (km/s)')
plt.ylabel('Prob. Density')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.ylim([0, 0.0035])
plt.xlim([-1000, 0])
plt.grid()

plt.subplot(2, 3, 2)
data = earthward_vs[:, 1]
loc, sigma = norm.fit(data)
print(loc, sigma)
# plot data
plt.hist(data, density=True, bins=50, alpha=0.6, label='Observed Shock Components', color='#8059e7ff')
# plot fit
x = np.linspace(min(data), max(data), 500)
y = norm.pdf(x, loc, sigma)
plt.plot(x, y, label='Gaussian Fit', c='red')
plt.xlabel('Earthward GSE Vy (km/s)')
plt.yticks(color='black')
plt.ylim([0, 0.0035])
plt.xlim([-500, 500])
plt.grid()

plt.subplot(2, 3, 3)
data = earthward_vs[:, 2]
loc, sigma = norm.fit(data)
print(loc, sigma)
# plot data
plt.hist(data, density=True, bins=50, alpha=0.6, label='Observed Shock Components', color='#8059e7ff')
# plot fit
x = np.linspace(min(data), max(data), 500)
y = norm.pdf(x, loc, sigma)
plt.plot(x, y, label='Gaussian Fit', c='red')
plt.xlabel('Earthward GSE Vz (km/s)')
plt.yticks(color='black')
plt.ylim([0, 0.0035])
plt.xlim([-500, 500])
plt.grid()

plt.subplot(2, 3, 4)
data = sunward_vs[:, 0]
loc, sigma = norm.fit(data)
print(loc, sigma)
# plot data
plt.hist(data, density=True, bins=50, alpha=0.6, label='Observed Shock Components', color='#8059e7ff')
# plot fit
x = np.linspace(min(data), max(data), 500)
y = norm.pdf(x, loc, sigma)
plt.plot(x, y, label='Gaussian Fit', c='red')
plt.xlabel('Sunward GSE Vx (km/s)')
plt.ylabel('Prob. Density')
plt.ylim([0, 0.0035])
plt.xlim([0, 1000])
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.grid()

plt.subplot(2, 3, 5)
data = sunward_vs[:, 1]
loc, sigma = norm.fit(data)
print(loc, sigma)
# plot data
plt.hist(data, density=True, bins=50, alpha=0.6, label='Observed Shock Components', color='#8059e7ff')
# plot fit
x = np.linspace(min(data), max(data), 500)
y = norm.pdf(x, loc, sigma)
plt.plot(x, y, label='Gaussian Fit', c='red')
plt.xlabel('Sunward GSE Vy (km/s)')
plt.yticks(color='black')
plt.ylim([0, 0.0035])
plt.xlim([-500, 500])
plt.grid()

plt.subplot(2, 3, 6)
data = sunward_vs[:, 2]
loc, sigma = norm.fit(data)
print(loc, sigma)
# plot data
plt.hist(data, density=True, bins=50, alpha=0.6, label='Observed Shock Components', color='#8059e7ff')
# plot fit
x = np.linspace(min(data), max(data), 500)
y = norm.pdf(x, loc, sigma)
plt.plot(x, y, label='Gaussian Fit', c='red')
plt.xlabel('Sunward GSE Vz (km/s)')
plt.yticks(color='black')
plt.ylim([0, 0.0035])
plt.xlim([-500, 500])
plt.grid()

# Get handles and labels from the first subplot
handles, labels = ax1.get_legend_handles_labels()
# Add legend for just the first subplot's items, positioned at figure level
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2)
plt.subplots_adjust(top=0.9)  # Adjust this value as needed
plt.tight_layout()

# plt.savefig('training_components_fig.png', dpi=300, format='png')

plt.show()

