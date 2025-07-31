import ctypes
import os
import numpy as np
from astropy import constants as const
from astropy import units as u
import re

################################################################################
# UNITS

# chosen units (same name as in astropy https://docs.astropy.org/en/stable/units/#module-astropy.units.astrophys)
my_u = ["pc","yr","M_sun"]

# physical constant converted to chosen units
G = const.G.to(my_u[0]+"^3/("+my_u[1]+"^2 "+my_u[2]+")").value
c = const.c.to(my_u[0]+"/"+my_u[1]).value
yr_in_h = 365.25*24
# print("G =",G,"c =",c)

################################################################################

# file with define statements
file_runtime_setup = "orbit/runtime_setup.h"

# compile the code, if needed, exit if compilation errors are raised
cmd = os.system("cd orbit && make && cd ..")
if cmd > 0:
    print("compilation error!")
    exit()

# Load the shared library
lib = ctypes.CDLL('orbit/libevolve.so')  # Use the correct path for your shared library

# Define the structure for Vector2DData
class Vector2DData(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),  # double**
        ("rows", ctypes.c_int),  # Number of rows
        ("cols", ctypes.c_int)   # Number of columns
    ]

# Define the argument and return types for the evolve function
lib.evolve.argtypes = [
    ctypes.c_char_p,                           # char path[]
    ctypes.c_char_p,                           # char file_name[]
    ctypes.c_double,                           # my_type t_end (assuming it's double)
    np.ctypeslib.ndpointer(dtype=np.float64),  # my_type *y_in (as double* in NumPy)
    np.ctypeslib.ndpointer(dtype=np.float64),  # my_type *params_in (as double* in NumPy)
    ctypes.c_int,                              # int N_y_in (size of y_in array)
    ctypes.c_int,                              # int N_params_in (size of params_in array)
    ctypes.c_int                               # int out_mode
]

# Define the return type of evolve (which is Vector2DData)
lib.evolve.restype = Vector2DData

# Load the free_vector2d function
lib.free_vector2d.argtypes = [Vector2DData]  # Takes a Vector2DData struct
lib.free_vector2d.restype = None  # Returns nothing

############################################################################################################

def process_vector2d_result(vector2d):
    rows, cols = vector2d.rows, vector2d.cols
    result = np.zeros((rows, cols))  # Create an empty NumPy array to hold the result

    # Iterate over the rows and copy data from C++ structure into the NumPy array
    for i in range(rows):
        row_ptr = ctypes.cast(vector2d.data[i], ctypes.POINTER(ctypes.c_double))  # Cast row to double pointer
        # print(f"Row {i} pointer: {row_ptr}")
        result[i, :] = np.ctypeslib.as_array(row_ptr, shape=(cols,))
        # print(f"Row {i} values: {result[i, :]}")
    
    return result

############################################################################################################

def extract_defines(file_path):
    defines = []  # List to store the valid #define statements
    
    # Regular expression for uncommented #define lines
    define_pattern = r'^\s*#define\s+(\S+)(.*)'  # Captures the macro and optional value
    
    with open(file_path, 'r') as file:
        for line in file:
            # Ignore lines that are commented out
            line = line.strip()
            if line.startswith('//') or line.startswith('/*') or '*/' in line:
                continue
            
            # Match uncommented #define statements
            match = re.match(define_pattern, line)
            if match:
                defines.append(line)  # Append the valid line
    
    return defines

############################################################################################################
############################################################################################################

# usage with geodesic
def call_evolve_geo(path_in, file_name_in, T, r,v, args, out_mode_in=3):
    # Define inputs
    path = path_in.encode('utf-8')         # Example path to the file (byte string for C++ char*)
    file_name = file_name_in.encode('utf-8')     # Example file name (byte string for C++ char*)
    t_end = T                # Example double value for t_end
    y_in = np.array([0,1, r[0],v[0], r[1],v[1], r[2],v[2]], dtype=np.float64)  # Example input array y_in
    params_in = np.array(args, dtype=np.float64)  # Example params_in
    N_y_in = len(y_in)           # Size of y_in
    N_params_in = len(params_in) # Size of params_in
    out_mode = int(out_mode_in)

    # print(type(N_y_in),type(out_mode))
    
    # Call the evolve function
    result_struct = lib.evolve(
        path,
        file_name,
        t_end,
        y_in,
        params_in,
        N_y_in,
        N_params_in,
        out_mode
    )

    # print(result_struct.rows, result_struct.cols, result_struct.data[0][0])

    # Process the result (convert the C++ double** data to a NumPy 2D array)
    result_array = process_vector2d_result(result_struct)

    # Free the memory after using the result
    lib.free_vector2d(result_struct)
    # print("Resulting 2D Array from C++:\n", result_array, result_array.T.shape)
   
    return result_array.T


# usage with geodesic
def call_evolve_PN(path_in, file_name_in, T, R,V,S,SIGMA, args, out_mode_in=3):
    # Define inputs
    path = path_in.encode('utf-8')         # Example path to the file (byte string for C++ char*)
    file_name = file_name_in.encode('utf-8')     # Example file name (byte string for C++ char*)
    t_end = T                # Example double value for t_end
    y_in = np.array([ R[0],V[0],R[1],V[1],R[2],V[2],  S[0],S[1],S[2],  SIGMA[0],SIGMA[1],SIGMA[2] ], dtype=np.float64)  # Example input array y_in
    params_in = np.array(args, dtype=np.float64)  # Example params_in
    N_y_in = len(y_in)           # Size of y_in
    N_params_in = len(params_in) # Size of params_in
    out_mode = int(out_mode_in)

    # print(type(N_y_in),type(out_mode))
    
    # Call the evolve function
    result_struct = lib.evolve(
        path,
        file_name,
        t_end,
        y_in,
        params_in,
        N_y_in,
        N_params_in,
        out_mode
    )

    # print(result_struct.rows, result_struct.cols, result_struct.data[0][0])

    # Process the result (convert the C++ double** data to a NumPy 2D array)
    result_array = process_vector2d_result(result_struct)

    # Free the memory after using the result
    lib.free_vector2d(result_struct)
    # print("Resulting 2D Array from C++:\n", result_array,result_array.T.shape)

    return result_array.T

############################################################################################################

# radius in KS coordinates, used for GEO motion
def r_KS(x,y,z,a):
    K = 0.5*(x**2+y**2+z**2 - a**2)

    K2_a2_z2 = np.sqrt( K*K + a**2*z**2 )

    return np.sqrt( K + K2_a2_z2 )

# rotation around x axis
def Rx(theta):
	return np.array([[1,0,0],[0,np.cos(theta),np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])

# rotation around y axis
def Ry(theta):
	return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])

# rotation around z axis
def Rz(theta):
	return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

# versor of a generic reference system rotated by angle inc and Omega
def basis(Omega,inc):
    u1 = np.array([np.cos(Omega),np.sin(Omega),0])
    u2 = np.array([-np.cos(inc)*np.sin(Omega),np.cos(inc)*np.cos(Omega),np.sin(inc)])
    u3 = np.array([np.sin(inc)*np.sin(Omega),-np.sin(inc)*np.cos(Omega),np.cos(inc)])
    return u1,u2,u3

# realtive position and velocity in 3d
def binary_pos_vel(a,e,f,omega,Omega,inc,m):

    u1,u2,u3 = basis(Omega,inc)

    p = (1-e**2)*a
    radius = p/(1+e*np.cos(f))

    r = radius*(u1 * np.cos(f+omega) + u2*np.sin(f+omega))
    v = np.sqrt(G*m/p) * ( -u1*( e*np.sin(omega) + np.sin(f+omega) ) + u2*( e*np.cos(omega) + np.cos(f+omega) ) )

    return r,v

# isco location as a function of spin parameter
def Risco_Rg(spin_a):
	Z1 = 1 + (1-spin_a*spin_a)**(1./3.)*((1+spin_a)**(1./3.)+(1-spin_a)**(1./3.))
	Z2 = np.sqrt(3.*spin_a*spin_a+Z1*Z1)
	return 3 + Z2 - np.sqrt((3-Z1)*(3+Z1+2*Z2))

# precession frequency for the rigidly precessing disc
def lense_thirring_prec_freq(M, spin_a, p): # M [Msun]

	Rg = G*M/c**2
	Rin = Risco_Rg(spin_a)*Rg
	Rout = 3*94.*(M/1e6)**(-2./3.)*Rg
	R_array = np.linspace(Rin, Rout, 1024, endpoint=False)
	L = lambda R : R**-p*(1.-np.sqrt(Rin/R))**p * np.sqrt(G*M*R)
	Omega_Lt = lambda R : 0.5*c**3 / (G*M) / ((R/Rg)**1.5+spin_a) * (4*spin_a*((R/Rg)**-1.5) - 3*(spin_a**2.)*((R/Rg)**-2.)) # [Hz]
	Int_top = lambda R : Omega_Lt(R) * L(R) * 2.*np.pi * R
	Int_bottom = lambda R : L(R) * 2.*np.pi * R
	Omega_p = np.trapz(Int_top(R_array), R_array)/np.trapz(Int_bottom(R_array), R_array)

	return Omega_p

############################################################################################################
############################################################################################################
############################################################################################################

def integration(
m1, # [Msun]
m2, # [Msun]
semi_major_axis, # [Rg]
eccentricity, # [-]
orbital_inclination, # [deg]
argument_pericentre, # [deg]
longitude_asc_node, # [deg]
true_anomaly, # [deg]
dimless_spin, # [-]
disc_inclination, # [deg]
Rdisc_min, # [Rg]
Rdisc_max, # [Rg]
D_obs, # [pc]
THETA_obs, # [deg]
PHI_obs, # [deg]
path = "", 
file_name = "test",
out_mode = 3, # control output options, keep to 3 for production runs
full_output = False # if True returns full information about crossings, otherwise only crossing times
):   
    
    '''
    Function that evolves the trajectory of an EMRI given initial conditions and returns the crossing times (plus location and velocity if required).

    Input parameters:
    Compulsory:
    m1: primary mass [Msun]
    m2: secondary mass [Msun]
    sma: semi-major axis [Rg]
    ecc: eccentricity [-]
    inc: orbital inclination [deg]
    omega: argument of pericentre [deg]
    Omega: longitude ascending node [deg]
    f: true anomaly [deg]
    chi: dimensionless spin parameter [-]
    inc_d: disc inclination [deg]
    Rdisc_min: lower edge of the disc size [Rg]
    Rdisc_max: upper edge of the disc size [Rg]
    D_obs: radial distance of the observer [pc]
    theta_obs: polar angle of the observer [deg]
    phi_obs: azimuthal angle of the observer [deg]
    
    Optional:
    path = "": path where to store output, if any
    file_name = test: name for output file, if any
    out_mode = 3: controls output options: - keep to 3 to return crossing points (use this for production runs) 
                                           - use 2 for printing a file with full trajectory
                                           - [DO NOT USE] use 1 to get trajectory saved into an 2d array 
    full_output = True: controls the output of the function. 

    Output parameters:
    crossings details
    - if full_output = False -> only crossing times (with delays added) 
    - if full_output = True -> crossing times, positions, velocities, delays
    '''
    ##############################################
    # check defines in runtime_setup.h
    define_statements = extract_defines(file_runtime_setup)

    # Print the extracted #define statements
    if "#define PN_motion" in define_statements:
        PN_motion = True
        geo_motion = False
    elif "#define GEO_motion" in define_statements:
        PN_motion = False
        geo_motion = True
    else:
        print("wrong trajectory in runtime_setup.h: choose PN_motion or GEO_motion!")
        exit()

    ##############################################
    # conversion constants between physical and internal units
    M0 = m1+m2 # Msun
    R0 = G*M0/c**2
    T0 = G*M0/c**3

    #######
    # orbit
    e_in = eccentricity
    a_in = semi_major_axis*R0

    Period = 2*np.pi*np.sqrt(a_in**3/(G*M0))
    T_fin = 200*Period/T0
    # print("Period =",Period*yr_in_h,T_fin*T0*yr_in_h,T_fin)
    
    argument_pericentre_rad, longitude_asc_node_rad, orbital_inclination_rad, true_anomaly_rad = argument_pericentre*np.pi/180, longitude_asc_node*np.pi/180, orbital_inclination*np.pi/180, true_anomaly*np.pi/180
    r_in, v_in = binary_pos_vel(a_in,e_in,true_anomaly_rad,argument_pericentre_rad,longitude_asc_node_rad,orbital_inclination_rad,M0)

    #######
    # disc 
    # OMEGA_LT = 2.*c**3*chi1/G/m1*(50)**(-3)*T0 #physical units
    OMEGA_LT = lense_thirring_prec_freq(m1, dimless_spin, 1.85) # physical units, p=3./5.
    INCLINATION_LT = disc_inclination/180.*np.pi
    RDISC_MIN = Rdisc_min
    RDISC_MAX = Rdisc_max*(m1/1e6)**(-2/3)

    if INCLINATION_LT == 0:
        OMEGA_LT = 0

    #######
    # observer
    D_obs_dimless = D_obs/R0
    THETA_obs_rad = THETA_obs*np.pi/180
    PHI_obs_rad = PHI_obs*np.pi/180

    if PN_motion:
        
        nu_in = m1*m2/M0**2

        #######
        # spin
        chi1, chi2 = dimless_spin,0.0

        s1 = G*m1**2*chi1
        s2 = G*m2**2*chi2

        s1_vec = np.array([0,0,s1])
        s2_vec = np.array([0,s2,0])

        s = s1_vec+s2_vec
        Sigma = (m1+m2)*(s2_vec/m2-s1_vec/m1)

        #######
        # dimless variables
        R = r_in / R0
        V = v_in / (R0/T0)
        S = s / (M0*R0**3/T0**2)
        SIGMA = Sigma / (M0*R0**3/T0**2)   

        #######
        # arguments to be passed
        #// M0,R0,T0,PN1,PN2,PN2_5,PN3,PN3_5,PN1_5_spin,nu,OMEGA_LT,INCLINATION_LT,D_obs,THETA_obs,PHI_obs
        PN_args = np.array([ M0,R0,T0,1,1,1,1,1,1,nu_in,OMEGA_LT*T0,INCLINATION_LT,RDISC_MIN,RDISC_MAX,D_obs_dimless,THETA_obs_rad,PHI_obs_rad ])

        #######
        # crossings
        tc,xc,yc,zc,vxc,vyc,vzc,DELTA_R,DELTA_S,DELTA_E = call_evolve_PN(path, file_name, T_fin, R,V,S,SIGMA, PN_args, out_mode)

    if geo_motion:
	
        #######
        # dimless variables
        m1_dimless = m1/M0
        m2_dimless = m2/M0
        R = r_in / R0 
        V = v_in / (R0/T0)

        #######
        # arguments to be passed
        #// M0,R0,T0,m1,m2,spin_par,OMEGA_LT,INCLINATION_LT,D_obs,THETA_obs,PHI_obs
        kerr_parameters = np.array([ M0,R0,T0,m1_dimless,m2_dimless,dimless_spin,OMEGA_LT*T0,INCLINATION_LT,RDISC_MIN,RDISC_MAX,D_obs_dimless,THETA_obs_rad,PHI_obs_rad ]) # mass and spin parameters

        #######
        # crossings
        tc,xc,yc,zc,vxc,vyc,vzc,DELTA_R,DELTA_S,DELTA_E = call_evolve_geo(path, file_name, T_fin, R,V, kerr_parameters, out_mode)

    #######################################################################################################################################
    #######################################################################################################################################

    tc,xc,yc,zc,vxc,vyc,vzc,DELTA_R,DELTA_S,DELTA_E = tc[tc>0],xc[tc>0],yc[tc>0],zc[tc>0],vxc[tc>0],vyc[tc>0],vzc[tc>0],DELTA_R[tc>0],DELTA_S[tc>0],DELTA_E[tc>0]
    DELTA_R -= D_obs_dimless # remove constant time shift robs/c i.e. the time travel from origin (i.e. the binary barycenter) to robs

	# convert in physical units
    tc,xc,yc,zc,vxc,vyc,vzc,DELTA_R,DELTA_S,DELTA_E = tc*T0,xc*R0,yc*R0,zc*R0,vxc*R0/T0,vyc*R0/T0,vzc*R0/T0,DELTA_R*T0,DELTA_S*T0,DELTA_E*T0

    Tobs_del = tc+DELTA_R+DELTA_S+DELTA_E

    if full_output:
        return Tobs_del*yr_in_h,tc,xc,yc,zc,vxc,vyc,vzc
    else:    
        return Tobs_del*yr_in_h
