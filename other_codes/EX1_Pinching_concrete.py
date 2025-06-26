# --------------------------------------------------------------------------------------------------
# Example SDOF.  dynamic eq ground motion
#			Di Trapani & Colajanni  & Liuzzo , 11_02_2025
#
#    ^Y
#    |
#    2       __ 
#    |          | 
#    |          |
#    |          |
#  (1)       LCol
#    |          |
#    |          |
#    |          |
#  1=11        _|_  -------->X
# 

# SET UP ----------------------------------------------------------------------------
# units: Newton, mm, sec



# define GEOMETRY -------------------------------------------------------------


import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt

# unità di misura: N, mm, sec

ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 3)



# nodal coordinates:
ops.node(1, 0, 0)
ops.node(11, 0, 0)		



# Single point constraints -- Boundary Conditions
# node     DX  DY   RZ
ops.fix(1, 1,   1,   1)

# equalDOF $rNodeTag $cNodeTag $dof2 $dof3 ...
ops.equalDOF(1, 11, 2, 3)

# define MATERIALS -------------------------------------------------------------

# Elastic-Perfectly Plastic Material

Fy=600000.0      		# Yielding force / case 1
My=2668932.97    	   	# Yielding moment / case 2
# My=326200000    		# Yielding moment / case 3

deltay=26.16    		# Yielding displacement / case 1 [mm]
thetay=0.0224    		# Yielding curvature / case 2
# thetay=0.0248    		# Yielding curvature / case 3

deltau=200.000    		# Ultimate displacement / case 1
thetau=250 		# Ultimate curvature / case 2
# thetau=0.1108    		# Ultimate curvature / case 3



# K=Fy/deltay            	# Stiffness


# STEEL
# Reinforcing steel
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

# ops.uniaxialMaterial ('ElasticPP', 'matTag', K, epsyP, epsyN, eps0)
ops.uniaxialMaterial ('ElasticPP', 1, k,  deltay,   -deltay,    0)

# MinMax Material to assign  ultimate rotation limit

# ops.uniaxialMaterial ('MinMax', matTag, otherTag, '-min', minStrain, '-max', maxStrain)
ops.uniaxialMaterial('MinMax', 2, 3, '-min', -2.5*du, '-max', 2.5*du) 



ops.geomTransf('Linear', 1)



# ops.element ('zeroLength', eleTag, jNode, '-mat', matTag1, matTag2, ... -dir, dir1, $dir2 ..
ops.element   ('zeroLength',   1,     1,     11,    '-mat',   2,     '-dir', 1)


# Define masses
g = 9.81e3 # mm/s^2

# weight of a storey
# Ws = -q*Lport

# weigth of each generic node
# Wnode = (Ws/2+P)

# mass of each generic node
# Mnode = Wnode/g
T=0.5
Mnode=k*T**2/(4*3.1415**2)





# define GRAVITY LOADS-------------------------------------------------------------
P=1
ops.timeSeries('Linear', 1 )
ops.pattern ('Plain', 1, 1)
ops.load(1, 0, 0, 0) 

# ------------------------------
# Start of analysis generation
# ------------------------------

# Create the system of equation, a sparse solver with partial pivoting
ops.system ('BandGeneral')

# Create the constraint handler, the transformation method
ops.constraints ('Transformation')

# Create the DOF numberer, the reverse Cuthill-McKee algorithm
ops.numberer ('RCM')

# Create the convergence test, the norm of the residual with a tolerance of  1e-12 and a max number of iterations of 10
ops.test ('NormDispIncr', 1.0e-13,  1000, 3)

# Create the solution algorithm, a Newton-Raphson algorithm
ops.algorithm ('Newton')

# Create the integration scheme, the LoadControl scheme using steps of 0.05
ops.integrator ('LoadControl', 0.1)

# Create the analysis object
ops.analysis ('Static')

# ------------------------------
# End of analysis generation
# ------------------------------


# ------------------------------
# Finally perform the analysis
# ------------------------------

# perform the gravity load analysis, requires 20 steps to reach the load level
ops.analyze (10)

# ------------------------------------------------- maintain constant gravity loads and reset time to zero
ops.loadConst('-time', 0)

print ("Model Built")

# Print out the state of elements 

# opsv.plot_model()
# opsv.plot_defo (sfac=500)
# opsv.section_force_diagram_2d('M', 5.e-5)
# opsv.section_force_diagram_2d('N', 1.e-3)