# Jan 22 2023 Harrison Denning
#Code to optomize the mass of a 10 strut truss, using code provided by the textbook


import numpy as np
from truss import truss
import truss_complex
from math import sin, cos, sqrt, pi
from scipy.optimize import minimize, approx_fprime #calling in a specific function (minimize) from optomize module of scipy
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

mass_array = []

# function to run the truss function but I only care about the mass
def f_for_minimize_only(A): 
    mass, stress = truss(A)
    global mass_array #store the mass after each iteration
    mass_array.append(mass)
    return mass
def f(A): 
    mass, stress = truss(A)
    global mass_array #store the mass after each iteration
    #mass_array.append(mass)
    return mass

def g(A): 
    mass, stress = truss(A)
    return -(np.square(stress) - np.square(np.array([25000,25000,25000,25000,25000,25000,25000,25000,75000,25000])))

def f_jax(A):
    mass, stress = truss_complex.truss(A)
    global mass_array
    #mass_array.append(mass)
    return mass

def g_jax(A):
    mass, stress = truss_complex.truss(A)
    return stress


def BarConstraints(A): #create the constrain equation for the max stress what can be experienced
    mass, stress = truss(A)
    return -(np.square(stress) - np.square(np.array([25000,25000,25000,25000,25000,25000,25000,25000,75000,25000]))) #square the values so that the code can handle both positive and negative stress


cross_Sections_cmplx = np.array([.2+complex(0,0),.2+complex(0,0),.2+complex(0,0),.2+complex(0,0),.2+complex(0,0),.2+complex(0,0),.2+complex(0,0),.2+complex(0,0),.2+complex(0,0),.2+complex(0,0)]) #an inital guess for the cross sectional areas of the truss bars
cross_Sections = np.array([.2,.2,.2,.2,.2,.2,.2,.2,.2,.2])
cross_Sections_jax = jnp.array([.2,.2,.2,.2,.2,.2,.2,.2,.2,.2])

#calculate derivative of mass/crosssectional Area 
def Forward_finite_Dif(x, f, h): # find the gradient with forward finite differencing 
    jacobian = []
    f_first = f(x) #evaluate function at inital values
    # print("f_first:", f_first)
    for j in range(len(x)):
        # print('j value:', j)
        x_diff = h*(1+abs(x[j])) #find how much you pertubate by
        x[j] = x[j] + x_diff # get the value at the pertubated point
        f_next = f(x) # evaluate at the pertubated point
        # print('f_next', f_next)
        jacobian.append((f_next - f_first) / x_diff) # output a column of the jacobian
        x[j] = x[j] - x_diff #remove the perturbation
    return jacobian

def Complex_Grad(x, f, h):
    jacobian = []
    for j in range(len(x)): 
        # print(x[j])
        x[j] = x[j]+complex(0,h)
        # print('complex num:', x[j] )
        f_next = f(x) # evaluate the func with the complex step added
        j_val = np.imag(f_next) / h
        x[j] = x[j] - complex(0,h) #remove the step you just took 
        jacobian.append(j_val)
        # print(x[j])
    return jacobian

def jax_jacob(cross_sec):
    cross_Sec = jnp.array([.2,.2,.2,.2,.2,.2,.2,.2,.2,.2])
    f_value = f_jax(cross_sec)
    J = jax.jacobian(f_jax)
    f_jac_jax = J(cross_sec)
    # print(f"Function Value: {f_value}")
    # print(f"JAX MASS {f_jac_jax}")
    return f_jac_jax

def jax_constraint(cross_sec):
    cross_Sec = jnp.array([.2,.2,.2,.2,.2,.2,.2,.2,.2,.2])
    f_value = g_jax(cross_sec)
    J = jax.jacobian(g_jax)
    g_jac_jax = J(cross_sec)
    # print(f"Function Value: {f_value}")
    print(f"JAX STRESS {g_jac_jax}")
    return g_jac_jax

# perfrom derivations with forward difference
Mass_deriv = Forward_finite_Dif(cross_Sections, f, 10e-4)
# print('FWD DIF MASS', Mass_deriv)

Stress_deriv = Forward_finite_Dif(cross_Sections, g, 10e-4)
print('FWD DIF STRESS', Stress_deriv)

Mass_deriv_complex = Complex_Grad(cross_Sections_cmplx, f, 10e-200)
# print('FWD DIF MASS', Mass_deriv)

Stress_deriv_complex = Complex_Grad(cross_Sections_cmplx, g, 10e-200)
print('complex STRESS', Stress_deriv)
jax_mass_deriv = jax_jacob(cross_Sections)
jax_stress_deriv = jax_constraint(cross_Sections)

fwd_mass_error =abs((np.linalg.norm(Mass_deriv) - jnp.linalg.norm(jax_mass_deriv))/jnp.linalg.norm(jax_mass_deriv))
fwd_Stress_error=abs((np.linalg.norm(Stress_deriv) - jnp.linalg.norm(jax_stress_deriv))/jnp.linalg.norm(jax_stress_deriv))
cmplx_mass_error =abs((np.linalg.norm(Mass_deriv_complex) - jnp.linalg.norm(jax_mass_deriv))/jnp.linalg.norm(jax_mass_deriv))
cmplx_Stress_error=abs((np.linalg.norm(Stress_deriv_complex) - jnp.linalg.norm(jax_stress_deriv))/jnp.linalg.norm(jax_stress_deriv))
print('Error of fwd Mass', fwd_mass_error)
print('Error of fwd Stress', fwd_Stress_error)
print('Error of Complex Mass', cmplx_mass_error)
print('Error of Complex Stress', cmplx_Stress_error)

#compare complex and fwd to jax to find the percent difference


#set up functions that do the same thing as above to pass into minimize
def fwd_mass(cross_sec):
    return Forward_finite_Dif(cross_sec, f, 10e-4)
def fwd_stress(cross_sec):
    return Forward_finite_Dif(cross_sec, g, 10e-4)
def cmplx_mass(cross_sec):
    complex_cross = [complex(term,0) for term in cross_sec]
    return Complex_Grad(complex_cross, f, 10e-200)
def cmplx_stress(cross_sec):
    complex_cross = [complex(term,0) for term in cross_sec]
    return Complex_Grad(complex_cross, g, 10e-200)



#optomize using scypy minium
theconstraints = {'type':'ineq','fun':BarConstraints, 'jac' : cmplx_stress} #define the constraint conditions
thebounds = ((.1,None),(.1,None),(.1,None),(.1,None),(.1,None),(.1,None),(.1,None),(.1,None),(.1,None),(.1,None)) #create the boundary space for each of the 10 bars
res = minimize(f_for_minimize_only, cross_Sections, constraints= theconstraints, bounds= thebounds, jac = cmplx_mass) #run the minimization
func_calls = len(mass_array) #count how many time the function was called by counting the number of iterations

# print('Xipy \'s derivatives:', approx_fprime(cross_Sections, g, h))

print(res)


print(func_calls)

x_graph = [*range(0,len(mass_array),1)]

print(x_graph)
# plt.figure()
# # plt.plot( x_graph, y_results)
print(mass_array)
plt.plot(x_graph,mass_array)
plt.title('Convergence of Truss Mass Optomization')
plt.xlabel('# of Function Calls')
plt.ylabel('Mass (lbs)')
plt.show()

# constraints = NonlinearConstraint(,jac = jacobian) # put the jacobians in the constraint
#minimize(constraints, bounds, method = SLSQP, jac = True, options = ) #inputting our own jacobian into the minimize function