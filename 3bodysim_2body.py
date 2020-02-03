import scipy as sci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

G = 6.67408e-11 # N-m2/kg2

m_nd = 1.989e+30 # kg # mass of the Sun
r_nd = 5.326e+12 # m # distance between starts in Alpha Centauri
v_nd = 30000 # m/s # relative velocity of earth around Sun
t_nd = 79.91*12*30*24*60*60*0.51 # s # orbital period of Alpha Centauri

K1 = G*t_nd*m_nd/(r_nd**2*v_nd)
K2 = v_nd*t_nd/r_nd

m1 = 1.1 # Alpha centauri A
m2 = 0.907 # Alpha centauri B

# initial position
r1 = [-0.5, 0, 0] 
r2 = [0.5, 0, 0]

# conversion of pos vec to arrays
r1 = sci.array(r1, dtype='float64')
r2 = sci.array(r2, dtype='float64')

# find com
r_com = (m1*r1 + m2*r2)/(m1 + m2)

# define initial vel
v1 = [0.01, 0.01, 0]
v2 = [-0.05, 0, -0.1]

# conversion of vel to arrays
v1 = sci.array(v1, dtype = 'float64')
v2 = sci.array(v2, dtype = 'float64')

# vel of COM
v_com = (m1*v1 + m2*v2)/(m1 + m2)

def TwoBodyEquations(w,t,G,m1,m2):
    r1=w[:3]
    r2=w[3:6]
    v1=w[6:9]
    v2=w[9:12]
    r=sci.linalg.norm(r2-r1) #Calculate magnitude or norm of vector
    dv1bydt=K1*m2*(r2-r1)/r**3
    dv2bydt=K1*m1*(r1-r2)/r**3
    dr1bydt=K2*v1
    dr2bydt=K2*v2
    r_derivs=sci.concatenate((dr1bydt,dr2bydt))
    derivs=sci.concatenate((r_derivs,dv1bydt,dv2bydt))
    return derivs

#Package initial parameters
init_params=sci.array([r1,r2,v1,v2]) #create array of initial params
init_params=init_params.flatten() #flatten array to make it 1D
time_span=sci.linspace(0,8,500) #8 orbital periods and 500 points
#Run the ODE solver
import scipy.integrate
two_body_sol=sci.integrate.odeint(TwoBodyEquations,init_params,time_span,args=(G,m1,m2))

r1_sol=two_body_sol[:,:3]
r2_sol=two_body_sol[:,3:6]

#Create figure
fig=plt.figure(figsize=(15,15))
#Create 3D axes
ax=fig.add_subplot(111,projection="3d")
#Plot the orbits
ax.plot(r1_sol[:,0],r1_sol[:,1],r1_sol[:,2],color="darkblue")
ax.plot(r2_sol[:,0],r2_sol[:,1],r2_sol[:,2],color="tab:red")
#Plot the final positions of the stars
ax.scatter(r1_sol[-1,0],r1_sol[-1,1],r1_sol[-1,2],color="darkblue",marker="o",s=100,label="Alpha Centauri A")
ax.scatter(r2_sol[-1,0],r2_sol[-1,1],r2_sol[-1,2],color="tab:red",marker="o",s=100,label="Alpha Centauri B")
#Add a few more bells and whistles
ax.set_xlabel("x-coordinate",fontsize=14)
ax.set_ylabel("y-coordinate",fontsize=14)
ax.set_zlabel("z-coordinate",fontsize=14)
ax.set_title("Visualization of orbits of stars in a two-body system\n",fontsize=14)
ax.legend(loc="upper left",fontsize=14)
