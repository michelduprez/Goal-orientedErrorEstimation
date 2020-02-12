# Copyright (C) 2018 Michel Duprez, Huu Phuoc Bui.
#
# This file is part of dwr-linear.

# dwr-linear is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# dwr-linear is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with stochastic-hyperelasticity. If not, see <http://www.gnu.org/licenses/>.

"""
Mesh : tongue_sagittal.xml
System : -div(sigma(u)+sigma_A) = 0
Boundary condition : Dirichlet on the basis
                    Neumann on the remaining parts
Activation : fiting of fonctions f_red, f_green, f_blue
Region of interest : (0.0,66.0)*(69.0,100.0)) + (76.0,96.0)*(82.0,100.0)
Adaptation of the mesh : DWR or uniform
Outputs files : output_tongue_adapt_DWR.txt
                output_tongue_adapt_uniform.txt
Outputs : exlicite error	
        eta
        sum eta_T
        exact interest error
        global error
"""

from __future__ import division# Allow the division of integer 
from dolfin import * 
import matplotlib.pyplot as plt
#parameters["plotting_backend"] = "matplotlib"
from mshr import *
import numpy as np
#set_log_level(TRACE)
parameters["allow_extrapolation"] = True # for the refinement of the mesh
parameters["refinement_algorithm"] = "plaza_with_parent_facets" # for the refinement of the mesh
#set_log_level(WARNING)


###################################
##### beginning of parameters #####
###################################


# Lame coefficients
E = 0.001152# MPa (Tracqui et Ohayon, 2004)) 
nu = 0.4#0.49
lmbda= E*nu/((1-2*nu)*(1+nu))
mu=E/(2*(1+nu))
epsil=0.1
beta = 1.0


# parameters of activation
T = 0.00002
beta = 1.0


# discretisation step of the fibers
delta_x = 0.1


# ball of influence for the computation of the activation 
ball = 15.0

# choice of space : degree of polynom in V
degreeInV = 2

# number af refining before to start
ref = 2


# maximum of iterations of adaptation
max_ite = 3



# refinement fraction in the dorfler strategy of DWR
refinement_fraction = 0.8


# Output
Output_latex = True
Plot = True
Save = True
# Move the mesh with the solution (applied displacement)
Move = False


#############################
##### end of parameters #####
#############################


# Function describing the 3 fibers
def f_red(x):
        return 0.00857611287978083*x**2 -1.2301159815168*x + 95.8172515164194

def f_green(x):
        return 0.00554620636697113*x**2 -0.883886841564548*x + 83.0994088499738

def f_blue(x):
        return 0.00287942275234755*x**2 -0.607773586408168*x + 73.5319763387087

# Derivative of the function describing the fibers
def f_red_d(x):
        return 2*0.00857611287978083*x -1.2301159815168

def f_green_d(x):
        return 2*0.00554620636697113*x -0.883886841564548

def f_blue_d(x):
        return 2*0.00287942275234755*x -0.607773586408168


# Vectors of those functions
Para = [ f_red, f_green, f_blue ]
Para_d = [ f_red_d, f_green_d, f_blue_d ]


# Discretization of the fibers
# pts_fX, pts_fY: vector direction in x, y of the fibers
def points(delta_x):
        pts_x = np.arange(70.0,130.0,delta_x)
        pts_X = np.concatenate((pts_x,pts_x,pts_x),axis=0) #Join a sequence of arrays along an existing axis.
        pts_Y = []
        pts_fX = []
        pts_fY = []
        for p in range(3):
                pts_Y = np.concatenate((pts_Y,Para[p](pts_x)),axis=0)
                
                pts_d = Para_d[p](pts_x)
                pts_norm = np.sqrt(pts_d**2 + 1.0)
                
                pts_fX = np.concatenate((pts_fX,1.0/pts_norm),axis=0)		
                pts_fY = np.concatenate((pts_fY,pts_d/pts_norm),axis=0)
                
        return pts_X, pts_Y, pts_fX, pts_fY


# Distance between two given points
def dist(x,y):
        xId = x*np.ones(len(pts_X))
        yId = y*np.ones(len(pts_Y))
        return np.sqrt((xId - pts_X)**2+(yId - pts_Y)**2)


# Create a DG0 function of the activation : new version
# DG02 is VectorFunctionSpace(mesh,'DG',0,2) # Create vector-valued finite element function space.
def activation(mesh,DG02):
        f = Function(DG02)
        x1 = 69.77
        y1 = 58.02
        x2 = 95.86
        y2 = 69
        x3 = 114
        y3 = 76
        a1 = (y2-y1)/(x2-x1)
        b1 = y2-a1*x2
        a2 = (y3-y2)/(x3-x2)
        b2 = y2-a2*x2
        for c in range(mesh.num_cells()):
                f.vector()[2*c] = 0.0
                f.vector()[2*c+1] = 0.0
                gx = Cell(mesh,c).midpoint().x()
                gy = Cell(mesh,c).midpoint().y()
                if gy < a1*gx + b1 and gy < a2*gx + b2: # the cell is on a side of two defined lines
                        distance = dist(gx,gy) # distances to all points of 3 fibers

                        if min(distance)<1e-6:
                                argmin_distance = distance.argmin()
                                f1 = pts_fX(argmin_distance)
                                f2 = pts_fY(argmin_distance)
                        else:
                                cond = distance <= ball # True/False array
                                f1 = np.sum(np.extract(cond,pts_fX/distance))
                                f2 = np.sum(np.extract(cond,pts_fY/distance))
                        norm = np.sqrt(f1**2+f2**2)
                        f.vector()[2*c] = f1/norm
                        f.vector()[2*c+1] = f2/norm
        return f

    
    
# explicite residual estimator 
def explicite_residual(mesh,u_h,DG0,h,n,eA):
        w = TestFunction(DG0)
        sigmaA = beta*T*outer(eA,eA)
        # interior residual
        residual1 = h**2*w*(div(sigma(u_h) + sigmaA))**2*dx
        
        # jump at interior faces
        residual2 = avg(w)*avg(h)*inner((sigma(u_h)('+')+ sigmaA('+'))*n('+')+(sigma(u_h)('-')+ sigmaA('-'))*n('-'),(sigma(u_h)('+')+ sigmaA('+'))*n('+')+(sigma(u_h)('-')+ sigmaA('-'))*n('-'))*dS # dS: interior facet
        
        # jump at Neumann boundaries
        residual3 = w*h*((sigma(u_h)+sigmaA)*n)**2*ds(0) # ds(0) : Neumann boundaries
        
        residual =  residual1 + residual2 + residual3
        cell_residual = Function(DG0)
        error_explicite = assemble(residual, tensor=cell_residual.vector()).get_local()
        return error_explicite
    

# Refinement of the mesh from the indicator
# eta: vector of eta_K for each element
def dorfler_marking(eta, mesh):
        # eta_ind is 2xn
        eta_ind = np.concatenate((np.array([eta]),np.array([range(len(eta))])),axis=0) #Join a sequence of arrays along an existing axis.
        eta_ind = eta_ind.T[np.lexsort(np.fliplr(eta_ind.T).T)].T
        eta_ind = eta_ind[:,::-1] # reverse column of the array
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers.set_all(False)
        ind=0
        while np.sum(eta_ind[0,:ind])<refinement_fraction*np.sum(eta):
                cell_markers[int(eta_ind[1,ind])]=True
                ind=ind+1
        return cell_markers


# Number of refienement when we compute the "exact" solution
iter_ref_exact = 1

# Computation on fine mesh and consider the solution as the exact solution!!
# N_exact0, N_exact1: number of cells (triangles)
def func_u_exact(mesh,domains,domain_dirich,eA,N_exact0, N_exact1):
        mesh_exact = Mesh(mesh)
        domains_exact = MeshFunction("size_t", mesh_exact, mesh.topology().dim())
        domain_dirich_exact = MeshFunction("size_t", mesh_exact, mesh_exact.topology().dim())
        for i in range(mesh_exact.num_cells()):
                domains_exact[i] = domains[i]
                domain_dirich_exact[i] = domain_dirich[i]
        DG02_exact = DG02
        eA_exact = eA
        for i in range(iter_ref_exact):
                mesh_exact_refined = refine(mesh_exact)
                domains_exact = adapt(domains_exact, mesh_exact_refined)
                domain_dirich_exact = adapt(domain_dirich_exact, mesh_exact_refined)
                DG02_exact = VectorFunctionSpace(mesh_exact_refined,'DG',0,2) # degree 0 (constant by element), 2 dimensions
                eA_exact = activation(mesh_exact_refined, DG02_exact)
                mesh_exact = mesh_exact_refined
 

        N_exact0 = N_exact1
        N_exact1 = mesh_exact.num_cells()
        print("Number of cells for exact solution :",mesh_exact.num_cells())	
        V_exact = VectorFunctionSpace(mesh_exact,'CG',degreeInV,2) # CG for Continuous Garlerkin (i.e. Lagrange element), 2 means 2D dimension

        # Initialize mesh function for boundary domains
        mesh_exact.init(1,2) # compute faces (i.e. 2) around edges (i.e. 1)
        boundaries_exact = MeshFunction("size_t", mesh_exact, mesh_exact.topology().dim() - 1)
        boundaries_exact.set_all(0)
        
        # Dirichlet boundaries marked as 1, Neumann marked as 0
        for i in range(mesh_exact.num_cells()):
                for f in facets(Cell(mesh_exact,i)):
                        if domain_dirich_exact[i]==1 and f.exterior():
                                boundaries_exact[f]=1

        # Initialize cell function for domains
        dx_exact = Measure("dx")(subdomain_data = domains_exact)
        ds_exact = Measure("ds")(subdomain_data = boundaries_exact) # exterior facet

        # Dirichlet boundary conditions
        bcs_exact = DirichletBC(V_exact, Constant((0.0, 0.0)), boundaries_exact,1) # 1 for Dirchlet boundaries (marked as 1)

        # Define variational problem
        u_trial_exact = TrialFunction(V_exact)
        v_exact = TestFunction(V_exact)
        a_exact = inner(sigma(u_trial_exact),epsilon(v_exact))*dx_exact
        L_exact = -beta*T*inner(outer(eA_exact,eA_exact),epsilon(v_exact))*dx_exact

        # Computed solution
        u_exact = Function(V_exact)

        # Solve equation a = L with respect to u
        problem = LinearVariationalProblem(a_exact, L_exact, u_exact,bcs_exact)
        solver = LinearVariationalSolver(problem)
        solver.parameters["linear_solver"] = "mumps"
        solver.solve()
        return  N_exact0, N_exact1, mesh_exact, V_exact, dx_exact, u_exact


# Strain tensor
def epsilon(u):
    return sym(grad(u))


# Stress tensor
def sigma(u):
    return lmbda*tr(sym(grad(u)))*Identity(2)+2.0*mu*sym(grad(u))


# Region of interest
class Omega(SubDomain):
    def inside(self, x, on_boundary):
        x1 = 63.26
        y1 = 70.54
        x2 = 103.69
        y2 = 87.26
        a = (y2-y1)/(x2-x1)
        b = y2-a*x2+7
        return True if x[1]>a*x[0]+b else False


# Dirichlet condition
class Dirich_int(SubDomain):
    def inside(self, x, on_boundary):
        return True if (between(x[0],(0.0,111.0)) and  between(x[1],(0.0,62.5))) else False


# Construction of the domains
omega = Omega() # domain of interest
dirich_int = Dirich_int()


# Construction of the discretization of the fibers
pts_X, pts_Y, pts_fX, pts_fY = points(delta_x)


# Initialization of array for the output
rEz_h = 1.0

num_cell_array = np.zeros(max_ite) # row vector of max_ite elems
explicite_array = np.zeros(max_ite)
sum_eta_T_array = np.zeros(max_ite)
eta_array = np.zeros(max_ite)
exact_interest_error_array = np.zeros(max_ite)
exact_interest_array = np.zeros(max_ite)
exact_interest_relative_error_array = np.zeros(max_ite)
exact_global_error_array = np.zeros(max_ite)
eta_h_effi_array = np.zeros(max_ite)
sum_eta_T_effi_array = np.zeros(max_ite)


###################### models


# Generate the inital mesh
mesh = Mesh("tongue_sagittal.xml")


# Initialize cell function for domains
domains = MeshFunction("size_t", mesh, mesh.topology().dim()) # Create MeshFunction of topological codimension 0 on given mesh.
domains.set_all(0)
omega.mark(domains, 1) # mark domain of intereset omega as 1


# Initialize cell function for boundaries
domain_dirich = MeshFunction("size_t",mesh, mesh.topology().dim())
domain_dirich.set_all(0)
dirich_int.mark(domain_dirich,1)


# Construction of the activation
DG02 = VectorFunctionSpace(mesh,'DG',0,2) # degree 0 (constant by element), 2 dimensions

# eA is a Function(DG02): vector fiber constant on each element, 2D
eA = activation(mesh,DG02)
DG0 = FunctionSpace(mesh,'DG',0)


# Refienement before to start
for i in range(ref):
    mesh_adapted = refine(mesh)
    domains = adapt(domains, mesh_adapted)
    domain_dirich = adapt(domain_dirich, mesh_adapted)
    DG02 = VectorFunctionSpace(mesh_adapted,'DG',0,2) # degree 0 (constant by element), 2 dimensions
    eA = activation(mesh_adapted, DG02)
    mesh = mesh_adapted


# Used to limit the number of cells to compute the "exact" solution
N_exact0 = 0 # number of cells (triangles)
N_exact1 = 0

init = True
ite = 0

# Adaptive or uniform refiement 
uniform = False

# Maximum number of cell allowed for the mesh of the "exact" solution
max_N_exact = 1e6

# Beginning of the adaptive algorithm
while ite < max_ite and (N_exact0 == 0 or 2*N_exact1 - N_exact0 < max_N_exact):
        print("##############################")
        print("ITERATION:",ite+1)
        if init == True:
                init = False
        else:
                if uniform == False:
                        mesh_refined = refine(mesh, cell_markers)
                else:
                        mesh_refined = refine(mesh)
                domains = adapt(domains, mesh_refined)
                domain_dirich = adapt(domain_dirich, mesh_refined)
                DG02 = VectorFunctionSpace(mesh_refined,'DG',0,2)
                eA = activation(mesh_refined, DG02)
                mesh = mesh_refined
        
        # Initialize mesh function for boundary domains
        boundaries =  MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)
        mesh.init(1,2) # compute faces (i.e. 2) around edges (i.e. 1)
        
        #mark facet of Dirichet boundary
        for i in range(mesh.num_cells()):
                for f in facets(Cell(mesh,i)): # for each facet of the triangle
                        if domain_dirich[i]==1 and f.exterior():
                                boundaries[f]=1

        # normal and size of the cells
        n = FacetNormal(mesh) 
        h = CellDiameter(mesh)
        print("Number of cells :",mesh.num_cells())	

        # Construction of the different spaces
        V = VectorFunctionSpace(mesh,'CG',degreeInV,2) # 2 for dim
        
        DG0 = FunctionSpace(mesh,'DG',0)
        V_star2 = VectorFunctionSpace(mesh, "CG",degreeInV+1, 2) # 2 for dim
        print("Num pts : ",V.dim()/2)

        # Initialize cell function for domains, dS (interior_facet), ds (exterior_facet).
        dx = Measure("dx")(subdomain_data = domains)
        dS = Measure("dS")(subdomain_data = domains) # interior facet

        # Initialize mesh function for boundary domains,  ds (exterior_facet).
        ds = Measure("ds")(subdomain_data = boundaries) # exterior facet

        # Dirichlet boundary conditions
        # Dirichlet boundary is marked as 1
        bcs = DirichletBC(V, Constant((0.0, 0.0)), boundaries,1) # DirichletBC(V,g,sub_domain marker, sub_domain index number, method="topological")
        
        ##---------------------------------------- primal problem
        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(sigma(u),epsilon(v))*dx # inner means inner product
        
        # activation
        # eA is a Function(DG02): vector fiber constant on each element, 2D
        L = -beta*T*inner(outer(eA,eA),epsilon(v))*dx # outer means outer product

        # Compute solution
        u_h = Function(V)
        # Solve equation a = L with respect to u and the given boundary conditions, 
        # such that the estimated error (measured in M) is less than tol
        problem = LinearVariationalProblem(a, L, u_h,bcs)
        solver = LinearVariationalSolver(problem)
        solver.parameters["linear_solver"] = "mumps"
        solver.solve()

        # 1) -------------------------------------------------------- Computation of explicite error (Babuska et al 1978)
        error_explicite = explicite_residual(mesh,u_h,DG0,h,n,eA)
        print('explicit residual error',sum(abs(error_explicite))**(0.5))

        ##------------------------------------------  Dual problem       
        # Define dual variational problem in P(k+1)
        u_star2 = TrialFunction(V_star2)
        v_star2 = TestFunction(V_star2)
        a_star2 = inner(epsilon(u_star2),sigma(v_star2))*dx 


        L_star2 = tr(sigma(v_star2))*dx(1) # integrate on the domain omega (regeion of interest) marked as 1

        bcs = DirichletBC(V_star2, Constant((0.0, 0.0)), boundaries,1)
        z = Function(V_star2)
        problem = LinearVariationalProblem(a_star2, L_star2, z, bcs)
        solver = LinearVariationalSolver(problem)
        solver.parameters["linear_solver"] = "mumps"
        solver.solve()
        
        #plot(z, title="Dual solution")
        #plt.show()

        # 2)----------- Computation of the estimator eta_h: the GLOBAL estimator: J(u)-J(u_h) =  r(z) = l(z) - a(u_h,z)
        sigma_u_h = sigma(u_h)
        sigmaA = beta*T*outer(eA,eA)

        # 
        rEz_h = assemble(inner(sigma_u_h+sigmaA,epsilon(z))*dx)
        
        print ('rEz_h: ', rEz_h)
        
        eta_h = abs(rEz_h)
        print("||r(E(z_h))||:",eta_h)

        # 3)------------- Computation of the LOCAL estimator eta_T
        diff = z - interpolate(z,V) # z in V_star2
        w = TestFunction(DG0)
        eta_T1 = w*inner(div(sigma_u_h + sigmaA),diff)*dx # interior resisdual 
        eta_T2 = -2.0*avg(w)*0.5*inner((sigma_u_h('+')+ sigmaA('+'))*n('+')+(sigma_u_h('-')+ sigmaA('-'))*n('-'),avg(diff))*dS # jump at interior facet
        eta_T3 = -w*inner((sigma_u_h+sigmaA)*n,diff)*ds(0) # jump at Neumann boundary (marked as 0), Dirichlet marked as 1
        
        eta_T = eta_T1 + eta_T2 + eta_T3
        
        eta = abs(assemble(eta_T).get_local()) # assemble for each element T (triangle)
        print ('eta: ', eta.shape)
        sum_eta_T = sum(eta)
        print("Sum |eta_T|: ",sum_eta_T)
        #eta2 = assemble(eta_T).get_local()
        #sum_eta_T2 = abs(sum(eta2))
        #print("|Sum eta_T|: ",sum_eta_T2)
        
        
        # computations

        # L_2 norm of u_h
        print('L_2 norm of u_h',assemble(inner(u_h,u_h)*dx)**(0.5))

        # Computation the interpolation of the exact solution in CG
        N_exact0, N_exact1, mesh_exact, V_exact, dx_exact, u_exact = func_u_exact(mesh,domains,domain_dirich,eA,N_exact0, N_exact1)
        
        # interpolate u_h onto the space of V_exact (fine mesh)
        Iu_h = interpolate(u_h,V_exact)

        # Norm of the exact solution
        print('L_2 norm of the exact solution',assemble(inner(u_exact,u_exact)*dx_exact)**(0.5))

        # global error
        global_error = abs(assemble(inner(u_exact-Iu_h,u_exact-Iu_h)*dx_exact)**(0.5))
        print('Global L_2 norm of displacement error',global_error)

        # Error of quantity of interest
        err = as_vector((u_exact[0]-Iu_h[0], u_exact[1]-Iu_h[1])) # as_vector is a fenics function

        # J(u) - J(u_h) = J(u-u_h) since J is linear
        exact_interest_error = abs(assemble(tr(sigma(err))*dx_exact(1))) # domain of interest omega marked as 1

        print('Error quantity of interest',exact_interest_error)

        # Exact quantity of interest
        exact_interest = abs(assemble(tr(sigma(u_exact))*dx_exact(1)))

        print('Exact quantity of interest',exact_interest)
        print('Relative error quantity of interest',exact_interest_error/exact_interest)

        # Effectivity index: eta_h / exact quantity of interest
        print('Eta_h / |J(u)-J(u_h)|',eta_h/exact_interest_error)

        # Effectivity index: sum(eta_T) / exact quantity of interest
        print('sum |Eta_T| / |J(u)-J(u_h)|',sum_eta_T/exact_interest_error)

        # Construction of the indicator for the refienement
        cell_markers = dorfler_marking(eta, mesh)

        # Update arrays
        num_cell_array[ite] = mesh.num_cells()
        explicite_array[ite] = sum(abs(error_explicite))**(0.5)
        sum_eta_T_array[ite] = sum_eta_T
        eta_array[ite] = eta_h
        exact_interest_error_array[ite] = exact_interest_error
        exact_interest_array[ite] = exact_interest
        exact_interest_relative_error_array[ite] = exact_interest_error/exact_interest
        exact_global_error_array[ite] = global_error
        eta_h_effi_array[ite] = eta_h/exact_interest_error
        sum_eta_T_effi_array[ite] = sum_eta_T/exact_interest_error
        ite = ite + 1



# Function used to write in the outputs files
def output_latex(f,A,B):
        for i in range(len(A)):
                f.write('(')
                f.write(str(A[i]))
                f.write(',')
                f.write(str(B[i]))
                f.write(')\n')
        f.write('\n')


# print the different arrays in the outputs files
if Output_latex == True:
        if uniform == True:
                f = open('output_tongue_uniform.txt','w')
                f.write('Refinement : Uniform \n')
        else:
                f = open('output_tongue_DWR.txt','w')
                f.write('Refinement : DWR \n')
                f.write('Refinement Fraction : ')
                f.write(str(refinement_fraction))
                f.write('\n')
        f.write('Degree of polynom for CG : ')
        f.write(str(degreeInV))
        f.write('\n')
        f.write('Quantity of interest : ')

        f.write('int_omega tr(sigma(u))')

        f.write('\n')
        f.write('\n')
        f.write('Exlicite error : \n')	
        output_latex(f,num_cell_array,explicite_array)
        f.write('Eta_h : \n')	
        output_latex(f,num_cell_array,eta_array)
        f.write('Sum eta_T : \n')	
        output_latex(f,num_cell_array,sum_eta_T_array)
        f.write('Exact error of the quantity of interest |J(u_h)-J(u)| : \n')	
        output_latex(f,num_cell_array,exact_interest_error_array)
        f.write('Exact quantity of interest |J(u)| : \n')	
        output_latex(f,num_cell_array,exact_interest_array)
        f.write('Exact relative error of the quantity of interest |J(u_h)-J(u)|/|J(u)| : \n')	
        output_latex(f,num_cell_array,exact_interest_relative_error_array)
        f.write('Global error : \n')	
        output_latex(f,num_cell_array,exact_global_error_array)
        f.write('Eta_h / |J(u_h)-J(u)| : \n')	
        output_latex(f,num_cell_array,eta_h_effi_array)
        f.write('Sum eta_T / |J(u_h)-J(u)| : \n')	
        output_latex(f,num_cell_array,sum_eta_T_effi_array)
        f.close()

# Move the mesh with the solution
if Move == True:
        ALE.move(mesh,u_h)


# Region of activation
act_norm = MeshFunction("size_t",mesh, mesh.topology().dim())
act_norm.set_all(0)
for i in range(mesh.num_cells()):
        if eA.vector()[2*i] != 0:
                act_norm[i] = 1
                domains[i]=2


# Computation of the last error map
error_explicite_map = MeshFunction("double", mesh, mesh.topology().dim())
for i in range(len(error_explicite)):
        error_explicite_map[i] = abs(error_explicite)[i]


# Computation of the last error map
error_map = MeshFunction("double", mesh, mesh.topology().dim())
for i in range(len(eta)):
        error_map[i] = abs(eta)[i]

error_map = Function(DG0)
for c in range(mesh.num_cells()):
                error_map.vector()[c]=abs(eta)[c]
print('max eta',max(abs(eta)))


# Print size and computation of percentage of displacement
xmin = 90.0
xmax = 90.0
ymin = 60.0
ymax = 60.0
for v in vertices(mesh): 
        x = v.point().x()
        y = v.point().y()
        if x < xmin:
                xmin = x
        if x > xmax:
                xmax = x
        if y < ymin:
                ymin = y
        if y > ymax:
                ymax = y
print("Size of the Tongue",xmax - xmin,ymax - ymin)
pourcent_move = 0.0
for v in range(mesh.num_vertices()):
        depl = ((u_h.vector()[2*v]**2+u_h.vector()[2*v+1]**2)/((xmax-xmin)**2+(ymax-ymin)**2))**0.5
        if depl > pourcent_move:
                pourcent_move = depl
print("Percentage of displacement",pourcent_move)


# construction of the eigenvalue of epsilon(u_h)
eps_eig_max = Function(DG0)
shear_strain = Function(DG0)
DG02 = VectorFunctionSpace(mesh,'DG',0,2) #VectorFunctionSpace(mesh, family, degree, dim=None, form_degree=None, constrained_domain=None, restriction=None)
eps_eig_vec = Function(DG02)
epsilo = epsilon(u_h)
eps11DG = project(epsilo[0,0],DG0)
eps12DG = project(epsilo[0,1],DG0)
eps21DG = project(epsilo[1,0],DG0)
eps22DG = project(epsilo[1,1],DG0)
max_eigen_value = 0.0
for c in range(mesh.num_cells()):
        eps11 = float(eps11DG.vector()[c])
        eps12 = float(eps12DG.vector()[c])
        eps21 = float(eps21DG.vector()[c])
        eps22 = float(eps22DG.vector()[c])
        A = np.matrix([[eps11,eps12],[eps21,eps22]])
        evals, evecs = np.linalg.eig(A)
        if abs(evals[0]) > abs(evals[1]):
                eps_eig_max.vector()[c] = abs(evals[0])
                eps_eig_vec.vector()[2*c] = evecs[0,0]
                eps_eig_vec.vector()[2*c+1] = evecs[1,0]
        else:
                eps_eig_max.vector()[c] = abs(evals[1])
                eps_eig_vec.vector()[2*c] = evecs[0,1]
                eps_eig_vec.vector()[2*c+1] = evecs[1,1]
        shear_strain.vector()[c] = 0.5*(max(evals[0],evals[1])-min(evals[0],evals[1]))
        if max(abs(evals[0]),abs(evals[1])) > max_eigen_value:
                max_eigen_value = max(abs(evals[0]),abs(evals[1]))

print("Maximum eigenvalue of eps(u) : ",max_eigen_value)

# Plot
if Plot == True:
        plt.figure()
        plot_mesh = plot(mesh,title='final mesh')
        plt.savefig('tongue_final_mesh.png')
                
        plt.figure()
        plot_activation = plot(eA,title='activation')
        plt.savefig('tongue_activation.png')

        plt.figure()
        plot_u_h_x = plot(u_h[0],title = 'First componente of u_h')
        plt.savefig('tongue_first_componente_u_h.png')

        plt.figure()
        plot_u_h_y = plot(u_h[1],title = 'Second componente of u_h')
        plt.savefig('tongue_second_componente_u_h.png')

        plt.figure()
        plot_u_h_L2 = plot((abs(u_h[0])**2+abs(u_h[1])**2)**0.5,title='L2 norm u_h sol')
        plt.savefig('tongue_L2_norm_aprox.png')

        z_h = z
        
        plt.figure()
        plot_z_h_L2 = plot((abs(z_h[0])**2+abs(z_h[1])**2)**0.5,title='L2 norm z_h sol')
        plt.savefig('tongue_L2_norm_z_h.png')

        plt.figure()	
        epsi = epsilon(u_h)
        plot_max_eig_eps_u_h = plot(eps_eig_max,title='max eigenvalue eps_u_h sol')
        plt.savefig('tongue_max_eig_eps_u_h.png')

        plt.figure()
        plot_shear_strain = plot(shear_strain,title='shear strain')
        plt.savefig('tongue_Shear_strain.png')

        plt.figure()
        plot_error_expl = plot(error_explicite_map,title='explicite error map')
        plt.savefig('tongue_error_map_expl.png')

        plt.figure()	
        plot_error = plot(error_map,title='error map')
        plt.savefig('tongue_error_map.png')

        plt.figure()
        plot_domain = plot(domains,title='region of interest')
        plt.savefig('tongue_domains.png')

        plt.figure()	
        plot_boundaries = plot(boundaries,title='boundaries condition')
        plt.savefig('tongue_boundaries.png')

        plt.figure()
        plot_act_domain = plot(act_norm,title='Region of activation')
        plt.savefig('tongue_domain_activation.png')

        #plot_eps_eig1 = plot(eps_eig1,scale=0.0,title = 'First eigenvalue of epsilon(u_h)')
        #plot_eps_eig_vec = plot(eps_eig_vec,title='First eigenvector of epsilon(u_h)')

        # show all plot
        plt.show()

        # Save files
        if Save == True:

                File_mesh = File("tongue_mesh.xml")
                File_mesh << mesh
                File_mesh = File("tongue_mesh.pvd")
                File_mesh << mesh
                File = File("tongue_boundaries.xml")
                File << boundaries














