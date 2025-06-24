




import openseespy.opensees as ops
# import opsvis as opsv
import matplotlib.pyplot as plt
import plot_settings as plt_set
import numpy as np
import pickle



def os_analysis(id_acc, T_1, q):
    
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
    acc_data = np.array(accs_data["acc"][f'{id_acc}']) * g #converting acceleration to mm/s²

    # take position of accs_data["spectr"]["T"] equal to T_1
    T_index = accs_data["spectr"]["T"].index(T_1)

    S_el = accs_data["spectr"][f'{id_acc}'][T_index]  # spectral acceleration for T_1


    # Scale factor for acceleration
    a_g = (k * dy * q) / (Mnode * S_el * g)


    # Damping
    xi = 5/100
    alphas = 2*xi*np.sqrt(k/Mnode)
    ops.rayleigh(alphas, 0, 0, 0)

    # Parameters for the analysis
    dt = 0.005
    step_dt = 1
    DtAnalysis = dt/step_dt
    TMaxAnalysis = len(acc_data) * dt
    n_steps = int(TMaxAnalysis/DtAnalysis)+1
    dof_analysis = 1


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
    D = []
    ok, step = 0, 0

    while ok == 0 and tCurrent < TMaxAnalysis:

        ok = ops.analyze(1, DtAnalysis)

        if ok != 0:
            ops.test('NormDispIncr', 1e-4,  100, 0)
            ops.algorithm('ModifiedNewton', '-initial')
            ok = ops.analyze(1, DtAnalysis)
            if ok == 0:
                ops.test('EnergyIncr', 1e-8, 15, 0)
                ops.algorithm('Newton')
        
        ops.reactions()

        tCurrent = ops.getTime()
        time.append(tCurrent)

        D.append(ops.nodeDisp(top_node,dof_analysis))
        step += 1

    # Maximum displacement and index
    d_max = max(D, key=abs)
    id_max = D.index(d_max)

    return acc_data[int(id_max/step_dt)] / g *a_g


def eval_spectrum(id_acc, q, **kwargs):
    Ts = kwargs.get('Ts', np.arange(0.1, 4, 0.05))
    return [os_analysis(id_acc, round(T, 2), q) for T in Ts]

if __name__ == "__main__":

    id_acc = 1
    q = 1

    # S_d = os_analysis(id_acc, T_1, q)
    # print(f"Spectral displacement for id_acc={id_acc}, T_1={T_1}, q={q}: S_d = {S_d:.4f} g")



    S_ds = eval_spectrum(id_acc, q)





    Ts = np.arange(0.1, 4, 0.05)
    fig = plt.figure(figsize=plt_set.fig_size)
    ax = fig.gca()

    ax.plot(Ts, S_ds, label=f'id_acc={id_acc}, q={q}', c = "midnightblue")
    ax.scatter(Ts, S_ds, color='red', s=10)
    ax.set_xlabel('T [s]'), ax.set_ylabel(r'$S_d$ [g]')
    ax.grid(alpha = 0.3)

    ax.set_xlim(0), ax.set_ylim(0)
    ax.set_title(f'Spectr. acc={id_acc}, q={q}')
    plt.tight_layout(), plt.savefig(f'./figs/spectral_displacement_id_acc_{id_acc}_q_{q}.png', dpi=150, bbox_inches='tight')




