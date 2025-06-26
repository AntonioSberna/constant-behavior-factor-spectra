# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 05:04:01 2024

@author: Fabio Di Trapani
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 18:37:24 2024

@author: Fabio Di Trapani
"""
import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt
import numpy as np

import EX1_Pinching_concrete



LunitTXT = 1
node = 11
dof = 1
Lref=1

# Parameters particular to the model
IDctrlNode = node  # Node where displacement is read for displacement control
IDctrlDOF = dof  # Degree of freedom of displacement read for displacement control
iDmax = [10, 25, 50, 100, 150, 200, 250, 300]

# Vector of displacement-cycle peaks, in terms of storey drift ratio
Dincr_steps = 1   # Displacement increment for pushover
Fact = Lref  # Scale drift ratio by storey height for displacement cycles
CycleType = 2  # You can do Full / Push / Half cycles with the proc




# Create load pattern for lateral pushover load
Load = 1  # Define the lateral load as a proportion of the weight
PushNode = node # Define nodes where lateral load is applied in static lateral analysis
ops.timeSeries('Linear', 200)
ops.pattern('Plain', 200, 200)
ops.load(PushNode, Load, 0, 0)


def generate_peaks(d_max, d_incr, cycle_type : int, fact, d_init) -> np.array:
    # generate incremental disps for Dmax
    # this proc creates a file which defines a vector then executes the file to return the vector of disp. increments
    # structure of function by Silvia Mazzoni, development by APS
    # input variables:
    #	d_max	    : peak displacement (can be + or negative)
    #	d_incr      : displacement increment
    #	cycle_type  : 0 Push (0->+peak), 1 Half (0->+peak->0), 2 Full (0->+peak->0->-peak->0)
    #	fact	    : scaling factor (optional, default=1)

    n_step = int(abs(d_max) / d_incr) + 1
    # Check type of cycle value
    if cycle_type > 2 or cycle_type < 0:
        raise ValueError("Type of analysis not yet considered")
    
    # Push
    steps = np.linspace(0, d_max, num=n_step) + d_init

    # Half
    if cycle_type >= 1:
        steps = np.concatenate((steps, np.linspace(d_max-d_incr, 0, num=n_step-1)))

    # Complete cycle      
    if cycle_type >= 2:
        steps = np.concatenate((steps, np.linspace(-d_incr, -d_max, num=n_step-1)))
        steps = np.concatenate((steps, np.linspace(-d_max+d_incr, 0, num=n_step-1)))
    
    return steps*fact




D,V=[],[]
D0 = 0
fmt1 = "%s Cyclic analysis: CtrlNode %.3i, dof %.1i, Disp=%.4f %s"
for Dmax in iDmax: #for each cycle
    steps = generate_peaks(Dmax, Dincr_steps, CycleType, Fact, 0)
    for d_step in steps:
        d_incr = d_step - D0
        ops.integrator("DisplacementControl", IDctrlNode, IDctrlDOF, d_incr)
        ops.analysis("Static")
        ok = ops.analyze(1)
        if ok != 0:
            ops.test("NormDispIncr", 1.0e-4, 2000, 0)
            ops.algorithm("Newton", "-initial")
            ok = ops.analyze(1)
        if ok != 0:
            ops.algorithm("Broyden", 8)
            ok = ops.analyze(1)
        if ok != 0:
            ops.algorithm("NewtonLineSearch", 0.8)  
            ok = ops.analyze(1)


        if ok != 0:
            break
        else:
            ops.test('NormDispIncr', 1.0e-12, 1000)
            ops.algorithm("Newton")

        currentDisp = ops.nodeDisp(IDctrlNode, IDctrlDOF)
        D.append(currentDisp)
        ops.reactions()
        V.append(ops.nodeReaction(1, 1)/-1e3)

        D0 = d_step #move to next step




#  #creates empry arrays for storing results
fig = plt.figure()
ax=fig.gca()
ax.plot(D,V, linewidth=2, color='red')
ax.axhline(0, color='grey', linewidth = 1, alpha=0.6), ax.axvline(0, color='grey', linewidth = 1, alpha=0.6)
# Adding labels and title
ax.set_xlabel('Displacement [mm]')
ax.set_ylabel('Base Reaction [kN]')
ax.set_title('Cyclic response')
# Adding grid
ax.grid(True)

plt.show()



# opsv.plot_defo (sfac=10)
# # opsv.section_force_diagram_2d('M', 5.e-6)
# # opsv.section_force_diagram_2d('V', 1.e-2)
# # opsv.section_force_diagram_2d('N', 1.e-3)
