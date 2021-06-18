import numpy as np
from scipy.integrate import solve_ivp
import random
from collections import deque
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as anim

class SIRmodel:
    """A class to store data for SIR modelling"""
    def __init__(self,R0=4.,t_recup=20.,population=2**10,t_final=160.):
        # from arguments
        self.population=population
        self.R0=R0
        self.t_recup=t_recup
        
        # derived parameters
        self.I0=1./self.population      # initial infected fraction
        self.t_final=t_final
        
        # hard-coded
        self.numpoints=200
        self.run()
        
    def run(self):
         # solve dynamics
        sol = solve_ivp(fun=lambda t, y: self.eqns(t,y),
                        t_span=[0,self.t_final],
                        y0=[1.-self.I0,self.I0,0.],
                        t_eval=np.linspace(0,self.t_final,self.numpoints),
                        dense_output=True)
        self.t=sol.t
        self.S,self.I,self.G=sol.y
        self.pop_susceptible=np.ceil(self.S*self.population)
        self.pop_recovered=np.floor(self.G*self.population)
        self.pop_infectious=self.population-self.pop_susceptible-self.pop_recovered
        
    def eqns(self,t,v):
        S,I,G=v
        return [-self.R0*S*I/self.t_recup,(self.R0*S-1.)*I/self.t_recup,I/self.t_recup]
    
    def plot(self,ax):
        # population curves
        ax.plot(self.t,self.S*100,color='xkcd:blue',label="susceptible")
        ax.plot(self.t,self.I*100,color='xkcd:red',label="infectieuse")
        ax.plot(self.t,self.G*100,color='xkcd:green',label="guérie")
        
        ax.set_xlim(0, self.t_final)
        ax.set_ylim(0,100)
        ax.set_xlabel('jours')
        ax.set_ylabel('% population')
        ax.grid()
        ax.legend()
                

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self,R0=4,temps_recup=20,temps_final=160,grid_width=50):
        ## parameters
        self.grid_width=grid_width        
        self.model=SIRmodel(population=self.grid_width**2,R0=R0,t_recup=temps_recup,t_final=temps_final)
        
        ## Plot
        plt.rcParams.update({'font.size': 20,'lines.markersize':9,'lines.linewidth':4,'legend.loc':'right'})
        self.fig,self.alist = plt.subplots(1,2,figsize=(19,10))
        # crop sides
        self.fig.subplots_adjust(left=0.06, bottom=0.08, right=0.99, top=0.95,wspace=None, hspace=None)

        self.model.plot(self.alist[0])
        self.alist[0].set_title('modèle SIR - R0={}, guérison {} jours'.format(self.model.R0,int(self.model.t_recup)))
        # population dots
        self.alist[1].set_xlim(-1,self.grid_width)
        self.alist[1].axis("off")
        self.alist[1].set_title('population de {} personnes'.format(self.model.population))
        # Then setup FuncAnimation.
        self.ani = anim.FuncAnimation(self.fig, self.update, interval=8,frames=self.model.numpoints,
                                      init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        ## initialisation
        self.susceptibles=deque([*range(0,self.model.population)])
        self.coordy,self.coordx=divmod(np.array(self.susceptibles),self.grid_width)
        #random.shuffle(self.susceptibles)
        self.infectious=deque()
        self.recovered=deque()
        self.color=[0 for k in range(0,self.model.population)]
        # infect 1 person
        self.infectious.appendleft(self.susceptibles[0])
        self.color[self.susceptibles[0]]=1
        self.susceptibles.popleft()
        
        """Initial drawing of the scatter plot."""
        self.scat = self.alist[1].scatter(self.coordx, self.coordy,
                                          c=self.color,
                                          cmap='brg', vmin=0,vmax=2)
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        self.timeline=self.alist[0].axvline(0,ls='-',color='k',zorder=10)
        return self.scat,self.timeline,

    def update(self, i):
        """Update the scatter plot."""
        nrecov=int(self.model.pop_recovered[i])-len(self.recovered)
        nsick=len(self.susceptibles)-int(self.model.pop_susceptible[i])
        if nrecov>0:
            for k in range(0,nrecov):
                # 1 person recovered from right of the infected queue
                self.recovered.appendleft(self.infectious[-1])
                self.color[self.infectious[-1]]=2
                self.infectious.pop()
        
        if nsick>0:
            for k in range(0,nsick):
                # 1 person recovered from right of the infected queue
                self.infectious.appendleft(self.susceptibles[0])
                self.color[self.susceptibles[0]]=1
                self.susceptibles.popleft()
        # Set colors..
        self.scat.set_array(np.array(self.color))

        self.timeline.set_xdata(self.model.t[i])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,self.timeline,


if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")

    try:
        R0=float(sys.argv[1])
    except:
        R0=4.
    try:
        temps_recup=float(sys.argv[2])
    except:
        temps_recup=20.

    try:
        temps_final=float(sys.argv[3])
    except:
        temps_final=160.
        
    a = AnimatedScatter(R0,temps_recup,temps_final)
    #a.ani.save('covid_model_R0{}-tc{}.mp4'.format(R0,temps_recup),fps=24,extra_args=['-vcodec','libx264'])
    plt.show()
