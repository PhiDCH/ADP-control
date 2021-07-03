import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from .visualize import Logger

class nonlinSystem():
    def __init__(self, dot_x, dimension):
        self.dot_x  = dot_x
        self.dimension = dimension      # (n_state, n_input)
        
    def setSimulationParam(self, max_step=1e-3, algo='RK45', t_sim=(0,10), x0=None, sample_time = 1e-2):
        # fixed step_size
        self.max_step = max_step
        self.algo = algo
        self.t_sim = t_sim
        if np.all(x0==None):
            self.x0 = np.ones((self.dimension[0],))
        else: self.x0 = x0
        self.sample_time = sample_time
        
    def integrate(self, x0, u, t_span, t_eval=None):      
        if len(np.array(u).shape)==1:
            u = np.expand_dims(u, axis=1)
        
        result = integrate.solve_ivp(fun=self.dot_x, args=(u,), y0=x0, t_span=t_span, t_eval=t_eval, method=self.algo, max_step=self.max_step, dense_output=True)
        return result.t, result.y.T
 
class controller():
    def __init__(self, system):
        self.system = system
        self.dot_x = self.system.dot_x
        self.logX = Logger('results')
        
    def setPolicyParam(self, q_func=None, R=None, phi_func=None, psi_func=None, u0=lambda x: 0, data_eval=0.1, num_data=10, explore_noise=lambda t: 2*np.sin(100*t)):
        if q_func==None:
            self.q_func = controller.q_func
        else:   self.q_func = q_func                # positive definite function
        if R==None:
            self.R = np.eye(self.system.dimension[1])
        else:   self.R = R                # symmetric and positive definite matix (1x(mxm))
        if phi_func==None:
            self.phi_func = controller.phi_func
        else:   self.phi_func = phi_func    # basis function for value function
        if psi_func==None:
            self.psi_func = controller.psi_func
        else:   self.psi_func = psi_func    # basis function for policy function        
        self.u0 = u0                
        self.data_eval = data_eval
        self.num_data = num_data
        self.explore_noise = explore_noise   
        
        self.logWa = Logger('results')   
        self.logWc = Logger('results')
    
    def step(self, dot_x, x0, t_span):
        result = integrate.solve_ivp(fun=dot_x, t_span=t_span, y0=x0, method=self.system.algo, max_step=self.system.max_step, dense_output=True)
        return result.t, result.y.T
      
    def offPolicy(self, stop_thres=1e-3, max_iter=30, visualize=True):
        self.visualize = visualize
        self.stop_thres = stop_thres
        self.max_iter = max_iter
        # collect data
        u = lambda t,x: self.u0(x) + self.explore_noise(t)
        dot_x = lambda t,x: self.dot_x(t, x, u(t,x))
        
        x_plot = [self.system.x0]
        t_plot = [self.system.t_sim[0]]
        
        dphi = [] 
        Iq = []
        Iupsi = []
        Ipsipsi = []
        for i in range(self.num_data):
            x_sample = [x_plot[-1]]
            t_sample = [t_plot[-1]]
            t_collect = t_plot[-1]
            while t_plot[-1] < t_collect + self.data_eval:
                t_span = (t_plot[-1], t_plot[-1] + self.system.sample_time)
                t_temp, x_temp = self.step(dot_x, x_plot[-1], t_span)
                if self.visualize:
                    self.logX.log('states_offPolicy', x_temp[-1], int(t_temp[-1]/self.system.sample_time))
                    
                t_sample.append(t_temp[-1])
                x_sample.append(x_temp[-1])
                
                x_plot.extend(x_temp[1:].tolist()),
                t_plot.extend(t_temp[1:].tolist())
                
            dphi_, Iq_, Iupsi_, Ipsipsi_ = self._getRowOffPolicyMatrix(np.array(t_sample), np.array(x_sample))
            dphi.append(dphi_)
            Iq.append(Iq_)
            Iupsi.append(Iupsi_)
            Ipsipsi.append(Ipsipsi_)

        # solve policy 
        save_Wc, save_Wa = self._policyEval(np.array(dphi), np.array(Iq), np.array(Iupsi), np.array(Ipsipsi))
        Waopt = save_Wa[-1]
        t_plot, x_plot = self._afterGainWopt(t_plot, x_plot, Waopt, 'states_offPolicy')

        return save_Wc[-1], save_Wa[-1]
        
    def _afterGainWopt(self, t_plot, x_plot, Waopt, section):
        u = lambda t,x: Waopt.dot(self.psi_func(x))
        dot_x = lambda t,x: self.dot_x(t, x, u(t,x))
        sample_time = self.system.sample_time
        start = t_plot[-1]
        stop = self.system.t_sim[1]
        N = int((stop - start)/sample_time)
        for i in range(N):
            t_temp, x_temp = self.step(dot_x, x_plot[-1], (t_plot[-1], t_plot[-1]+sample_time))
            if self.visualize:
                self.logX.log(section, x_temp[-1], int(t_temp[-1]/self.system.sample_time))
            t_plot.extend(t_temp[1:].tolist())
            x_plot.extend(x_temp[1:].tolist())
        
        return t_plot, x_plot
            
    def _policyEval(self, dphi, Iq, Iupsi, Ipsipsi):
        n_psi = len(self.psi_func(self.system.x0))
        n_phi = len(self.phi_func(self.system.x0))
        n_input = self.system.dimension[1]
        Wa = np.zeros((n_input, n_psi))
        
        save_Wc = []
        save_Wa = [Wa]
        for i in range(self.max_iter): 
            temp = -2*(Iupsi - Ipsipsi.dot(np.kron(Wa.T, np.eye(n_psi))))
            A = np.hstack((dphi, temp))
            B = Iq + Ipsipsi.dot(Wa.T.dot(self.R.dot(Wa)).flatten()) 
            Wca = np.linalg.pinv(A).dot(B)
            Wc = Wca[:n_phi]
            if self.visualize:
                self.logWc.log('offPolicy_Wc', Wc, i)
                self.logWa.log('offPolicy_Wa', Wa, i)
            try: 
                err = np.linalg.norm(save_Wc[-1] - save_Wc[-2], ord=2)
                if err < self.stop_thres:
                    break
            except: pass
            Wa = Wca[n_phi:].reshape((n_input, n_psi)).T.dot(np.linalg.inv(self.R)).T
            save_Wc.append(Wc)
            save_Wa.append(Wa)
        
        return save_Wc, save_Wa
        
    def _getRowOffPolicyMatrix(self, t_sample, x_sample):
        dphi_ = self.phi_func(x_sample[-1]) - self.phi_func(x_sample[0])
        Iq_ = []
        Iupsi_ = []
        Ipsipsi_ = []
        u = lambda t,x: self.u0(x) + self.explore_noise(t)
        for i, xi in enumerate(x_sample):
            Iq_.append(self.q_func(xi))
            Iupsi_.append(np.kron(u(t_sample[i], xi), self.psi_func(xi)))
            Ipsipsi_.append(np.kron(self.psi_func(xi), self.psi_func(xi)))
            
        Iq_ = integrate.simpson(Iq_, t_sample, axis=0)
        Iupsi_ = integrate.simpson(Iupsi_, t_sample, axis=0)
        Ipsipsi_ = integrate.simpson(Ipsipsi_, t_sample, axis=0)
        return dphi_, Iq_, Iupsi_, Ipsipsi_
        
               
    @staticmethod    
    def psi_func(x):
        psi = []
        for i in range(len(x)):
            psi.append(x[i])
            for j in range(i, len(x)):
                for k in range(j, len(x)):
                    psi.append(x[i]*x[j]*x[k])
        return np.array(psi)
    @staticmethod
    def phi_func(x):
        phi = []
        for i in range(len(x)):
            for j in range(i, len(x)):
                phi.append(x[i]*x[j])
                phi.append(x[i]**2*x[j]**2)
        return np.array(phi)
    @staticmethod
    def q_func(x):
        return np.sum(x*x, axis=0)
 
 
 
                
def dot_x(t,x,u):
    # Dynamics of the syspension system
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]

    # Coefficients
    mb = 300    # kg
    mw = 60     # kg
    bs = 1000   # N/m/s
    ks = 16000 # N/m
    kt = 190000 # N/m
    kn = ks/10  # N/m

    # System Dynamics
    dx1 = x2
    dx2 = -(ks*(x1-x3)+kn*(x1-x3)**3)/mb - (bs*(x2-x4)-10000*u)/mb
    dx3 = x4
    dx4 =  (ks*(x1-x3)+kn*(x1-x3)**3)/mw + (bs*(x2-x4)-kt*x3-10000*u)/mw

    # Combine the output
    dx = [dx1, dx2, dx3, dx4]
    return dx       

if __name__ == '__main__':
    dimension = (4,1)
    system = nonlinSystem(dot_x, dimension)
    
    t_sim = (0,5)
    algo = 'RK45'
    max_step = 1e-3
    sample_time = 1e-3
    x0 = [0.1,-5,0.2,2]
    system.setSimulationParam(max_step=max_step, algo=algo, t_sim=t_sim, x0=x0, sample_time=sample_time)
    
    Controller = controller(system)
    u0 = 0
    data_eval = 0.01
    num_data = 80       # at leats n_phi+n_psi
    explore_noise = lambda t: 0.2*np.sum(np.sin(np.array([1, 3, 7, 11, 13, 15])*t))
    Controller.setPolicyParam(data_eval=data_eval, num_data=num_data, explore_noise=explore_noise)
    Wc, Wa = Controller.offPolicy()
    print(Wc)
    print(Wa)
    