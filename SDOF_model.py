




import openseespy.opensees as ops
# import opsvis as opsv
import matplotlib.pyplot as plt
import numpy as np
import pickle


id_acc = 1
T_1 = 0.5
q = 1


# unit of measurement: N, mm, sec

ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 3)

# Nodes
ops.node(1, 0, 0), ops.fix(1, 1, 1, 1)
ops.node(11, 0, 0), ops.equalDOF(1, 11, 2, 3)


# Material
fy = 57382058.84  # Yield stress [N]
fmaxfy=1.13
fresfy=0
k = 2193503.77  # Young's modulus [N/mm]
dy=fy/k
dp=dy*1.3
thetap= 60.2
dpc= 183
du=350


ops.uniaxialMaterial ('IMKPinching', 3, k, dp, dpc, du, fy, fmaxfy, fresfy, -dp, -dpc, -du, -fy, fmaxfy, fresfy, 400, 1000, 700, 400, 1, 1, 1, 1, 1, 1, 0.5, 0.5)
ops.uniaxialMaterial('MinMax', 2, 3, '-min', -2.5*du, '-max', 2.5*du) 


# Element
ops.geomTransf('Linear', 1)
ops.element('zeroLength', 1, 1, 11, '-mat', 2, '-dir', 1)




# One step of static analysis
ops.timeSeries('Linear', 1 )
ops.pattern ('Plain', 1, 1)
ops.load(1, 0, 0, 0) 


ops.system ('BandGeneral')
ops.constraints ('Transformation')
ops.numberer ('RCM')
ops.test ('NormDispIncr', 1.0e-13,  1000, 3)
ops.algorithm ('Newton')
ops.integrator ('LoadControl', 1)
ops.analysis ('Static')

ops.analyze(1)
ops.loadConst('-time', 0)


ops.wipeAnalysis()




#
#  NLTH analysis
#
g = 9806.65  # mm/s²

# Mass
Mnode = k*T_1**2/(4*np.pi**2)
ops.mass(11, Mnode, 0, 0)

## Modal analysis
# eigenvalues = ops.eigen('-fullGenLapack',1)
# periods = [(2*np.pi)/np.sqrt(lam) for lam in eigenvalues]
# for i, freq in enumerate(periods):
#     print(f"Modo {i+1}: frequenza = {freq:.4f} sec")



# read acceleration data from pickle file
with open('acc_data.pkl', 'rb') as f:
    accs_data = pickle.load(f)
acc_data = np.array(accs_data[f'{id_acc}']) * g #converting acceleration to mm/s²

# Scale factor for acceleration
a_g = (k * dy * q) / (Mnode * )
print("A_g", a_g)

# Damping
xi = 5/100
alphas = 2*xi*np.sqrt(k/Mnode)
ops.rayleigh(alphas, 0, 0, 0)

# Parameters for the analysis
dt = 0.005
DtAnalysis = dt/2
TMaxAnalysis = len(acc_data) * dt
n_steps = int(TMaxAnalysis/DtAnalysis)+1
dof_analysis = 1

print(TMaxAnalysis)

ops.test('EnergyIncr', 1e-8, 15, 0)
ops.constraints('Transformation')
ops.numberer('RCM')
ops.system("BandGen")
ops.algorithm('Newton')
ops.integrator('Newmark', 0.5, 0.25)
ops.analysis('Transient')

ops.timeSeries('Path', 2, '-dt', dt, '-values', *acc_data, '-factor', a_g)
ops.pattern('UniformExcitation', 10, dof_analysis, '-accel', 2)



top_node, basenodes = 11, [1]

tCurrent = ops.getTime()
time = []
D,R = [], []
ok, step = 0, 0

while ok == 0 and tCurrent < TMaxAnalysis:

    ok = ops.analyze(1, DtAnalysis)

    if ok != 0:
        print("regular newton failed .. lets try an initail stiffness for this step")
        ops.test('NormDispIncr', 1e-4,  100, 0)
        ops.algorithm('ModifiedNewton', '-initial')
        ok = ops.analyze(1, DtAnalysis)
        if ok == 0:
            print("that worked .. back to regular newton")
            ops.test('EnergyIncr', 1e-8, 15, 0)
            ops.algorithm('Newton')
    
    ops.reactions()

    tCurrent = ops.getTime()
    time.append(tCurrent)

    D.append(ops.nodeDisp(top_node,dof_analysis))
    R.append(-sum([ops.nodeReaction(node, dof_analysis)/1000 for node in basenodes]))
    step += 1

# Maximum displacement and index
d_max = max(D, key=abs)
id_max = D.index(d_max)

print(f"Maximum displacement: {d_max:.2f} mm at time {time[id_max]:.2f} s, Reaction: {R[id_max]:.2f} kN")


# Some plots
fig = plt.figure()
ax = fig.gca()
ax.plot(D,R, c = "midnightblue")
ax.scatter(d_max, R[id_max], color='red', label=f'Max: {d_max:.2f} mm at {time[id_max]:.2f} s')
ax.grid(alpha = 0.25)
ax.set_xlabel("Displacement [mm]")
ax.set_ylabel("Force [kN]")
ax.axhline(0, color='grey', linewidth = 1, alpha=0.6), ax.axvline(0, color='grey', linewidth = 1, alpha=0.6)




fig = plt.figure()
ax = fig.gca()
ax.plot(time, D, c = "midnightblue")
ax.scatter(time[id_max], d_max, color='red', label=f'Max: {d_max:.2f} mm at {time[id_max]:.2f} s')
ax.grid(alpha = 0.25)
ax.set_ylabel("Displacement [mm]")
ax.set_xlabel("Time [s]")
ax.axhline(0, color='grey', linewidth = 1, alpha=0.6), ax.axvline(0, color='grey', linewidth = 1, alpha=0.6)

fig = plt.figure()
ax = fig.gca()
ax.plot(time, acc_data, c = "midnightblue")
ax.grid(alpha = 0.25)
ax.set_ylabel("Acceleration [m/s²]")
ax.set_xlabel("Time [s]")
ax.axhline(0, color='grey', linewidth = 1, alpha=0.6), ax.axvline(0, color='grey', linewidth = 1, alpha=0.6)



plt.show()
