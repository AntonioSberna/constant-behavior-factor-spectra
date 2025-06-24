

import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt
import numpy as np


import EX1_Pinching_concrete


ops.loadConst('-time', 0)


# Wipe previous analysis
ops.wipeAnalysis()

# Define the dynamic analysis parameters
dt = 0.01
DtAnalysis = dt/2
TMaxAnalysis = 30
n_steps = int(TMaxAnalysis/DtAnalysis)+1


dof_analysis = 1


top_node = 11
basenodes = [1]

# Convergence Test: tolerance
Tol = 1.e-8
# Convergence Test: maximum number of iterations that will be performed before "failure to converge" is returned
maxNumIter = 15
# Convergence Test: flag used to print information on convergence (optional)        # 1: print information on each step; 
printFlag = 0
# Convergence Test: type
TestType = 'EnergyIncr'

# Define Convergence Test
ops.test(TestType, Tol, maxNumIter, printFlag)

# Constrain handle
ops.constraints('Transformation')

# Create the DOF numberer
ops.numberer('RCM')

# System
ops.system("BandGen")

# Solution algorithm
ops.algorithm('Newton')

# Integrator
ops.integrator('Newmark', 0.5, 0.25)

# Analysis
ops.analysis('Transient')
k=20000
T=0.5
Mnode=k*T**2/(4*3.1415**2)
ops.mass(11, Mnode, 0, 0)

print (f"Massa Nodo = {Mnode:.1f}")
## DAMPING SDOF
xi = 5/100
Ts=2*3.1415*(Mnode/k)**.5
alphas=2*xi*(k/Mnode)**.5
ops.rayleigh(alphas, 0, 0, 0)


## Damping
##

# def rayleigh_damping(T:list, T_ind:list, xis):
    # omegas = 2*np.pi/T

    # omega_1 = omegas[T_ind[0]]
    # omega_2 = omegas[T_ind[1]]


    # mat = np.array([[1/omega_1, omega_1], [1/omega_2, omega_2]])

    
    # return np.dot(2*np.linalg.inv(mat), np.array(xis).T)


# Vibration periods
# Ts = (2*np.pi)/np.sqrt(ops.eigen(4))
# Periods to consider
# T_ind = [1, 3]
# Damping ratio
# xi = 2/100

# Rayleigh damping parameters
# alphas = rayleigh_damping(Ts, T_ind, [xi]*2)

# Define Rayleigh damping
# ops.rayleigh(alphas[0], 0, alphas[1], 0)


# Time Series
ops.timeSeries('Path', 2, '-dt', dt, '-filePath', "acc_1.acc", '-factor', 10)
# Pattern
ops.pattern('UniformExcitation', 10, 1, '-accel', 2)


#Set some parameters
tCurrent = ops.getTime()
time = [tCurrent]
D,R = [], []
ok = 0


el_tags = ops.getEleTags()
nels = len(el_tags)


Eds = np.zeros((n_steps, nels, 6))
step = 0
while ok == 0 and tCurrent < TMaxAnalysis:
    
    ok = ops.analyze(1, DtAnalysis)
    
    if ok != 0:
        print("regular newton failed .. lets try an initail stiffness for this step")
        ops.test('NormDispIncr', 1.0e-4,  100, 0)
        ops.algorithm('ModifiedNewton', '-initial')
        ok = ops.analyze(1, DtAnalysis)
        if ok == 0:
            print("that worked .. back to regular newton")
            ops.test(TestType, Tol, maxNumIter, printFlag)
            ops.algorithm('Newton')
    
    tCurrent = ops.getTime()
    time.append(tCurrent)

    D.append(ops.nodeDisp(11,dof_analysis))
    ops.reactions()
    
    R.append(-sum([ops.nodeReaction(node, dof_analysis)/1000 for node in basenodes]))
    # for el_i, ele_tag in enumerate(el_tags):
    #     nd1, nd2 = ops.eleNodes(ele_tag)
    #     Eds[step, el_i, :] = [ops.nodeDisp(nd1)[0],
    #                             ops.nodeDisp(nd1)[1],
    #                             ops.nodeDisp(nd1)[2],
    #                             ops.nodeDisp(nd2)[0],
    #                             ops.nodeDisp(nd2)[1],
    #                             ops.nodeDisp(nd2)[2]]
    step += 1


# StressStrain = np.array(StressStrain)
# fig = plt.figure()
# ax = fig.gca()
# ax.plot(StressStrain[:,1]*100, StressStrain[:,0], c = "midnightblue")
# ax.grid(alpha = 0.25)
# ax.set_xlabel("Strain [%]")
# ax.set_ylabel("Stress [MPa]")
# ax.axhline(0, color='grey', linewidth = 1, alpha=0.6), ax.axvline(0, color='grey', linewidth = 1, alpha=0.6)
# fig.savefig("fig/dr.png", dpi = 300)



fig = plt.figure()
ax = fig.gca()
ax.plot(time[1:], D, c = "midnightblue")
ax.grid(alpha = 0.25)
ax.set_xlabel("Time [sec]")
ax.set_ylabel("Top displacement [mm]")
ax.set_xlim(0)
ax.axhline(0, color='grey', linewidth = 1, alpha=0.6), ax.axvline(0, color='grey', linewidth = 1, alpha=0.6)
# fig.savefig("fig/d_time.png", dpi = 300)


fig = plt.figure()
ax = fig.gca()
ax.plot(D,R, c = "midnightblue")
ax.grid(alpha = 0.25)
ax.set_xlabel("Displacement")
ax.set_ylabel("Force [mm]")
# ax.set_xlim(0)
ax.axhline(0, color='grey', linewidth = 1, alpha=0.6), ax.axvline(0, color='grey', linewidth = 1, alpha=0.6)
# fig.savefig("fig/d_time.png", dpi = 300)

plt.show()


# fmt_defo = {'color': 'blue', 'linestyle': 'solid', 'linewidth': 3.0,
#             'marker': '', 'markersize': 6}

# anim = opsv.anim_defo(Eds, time, 20, fmt_defo=fmt_defo,
#                     xlim=[-1e3, 8e3], ylim=[0, 1e4], fig_wi_he=(15., 20.))
# # anim.save("fig/anim_TH.gif", writer='pillow', fps=30)


Dmax=np.max(np.abs(D))
print (f"Max Displacement (mm)= {Dmax:.2f}")