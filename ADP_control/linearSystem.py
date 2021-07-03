import numpy as np  
from scipy import integrate
from control import lqr
from matplotlib import pyplot as plt 
from .visualize import Logger

class linearSystem():
    """
    This class present state-space linear system.
    
    Attributes: 
        dimension (tuple): (n_state, n_input).
        model (dict): {A, B, C, D, dimension}.
    
    """   
    def __init__(self, A, B, C=1, D=0):
        """The representation of a linear system in the form of state-space model

        Args:
            A (nxn array): the state matrix A
            B (nxm array): the input matrix B
            C (array, optional): the state output matrix C. Defaults to 1.
            D (array, optional): the input output matrix D. Defaults to 0.
            
        Note:
            The A, B matrix must be initialized with compatible dimension.
            
        """
        self.A = A
        self.B = B
        if len(self.B.shape)==1:
            self.B = np.expand_dims(self.B, axis=1)
            
        self.model_valid, self.dimension = self._check_model()
        self.C = C
        self.D = D
        self.model = {'A': self.A, 'B': self.B, 'C': C, 'D': D, 'dimension': self.dimension}
        
    def _check_model(self):          
        dimension = []
        model_valid = False
        a1,a2 = self.A.shape
        if a1 != a2:
            return model_valid
        else: n_states = a1
        
        b1,b2 = self.B.shape
        if b1 != a1:
            return model_valid 
        else: 
            n_inputs = b2
            model_valid=True
        dimension = [n_states, n_inputs]
        return model_valid, dimension

    def setSimulationParam(self, max_step=1e-3, algo='RK45', t_sim=(0,10), x0=None, sample_time = 1e-2):
        # fixed step_size
        """Run this function before any simulations

        Args:
            max_step (float, optional): define max step for ODEs solver algorithms. Defaults to 1e-3.
            algo (str, optional): RK45, RK23 or DOP853 . Defaults to 'RK45'.
            t_sim (tuple, optional): time for simualtion (start, stop). Defaults to (0,10).
            x0 (1xn array, optional): the initial state. Defaults to np.ones((n,)).
            sample_time (float, optional): the sample time. Defaults to 1e-2.
        """
        self.max_step = max_step
        self.algo = algo
        self.t_sim = t_sim
        if np.all(x0==None):
            self.x0 = np.ones((self.dimension[0],))
        else: self.x0 = x0
        self.sample_time = sample_time
       
    def integrate(self, x0, u, t_span):      
        if len(u.shape)==1:
            u = np.expand_dims(u, axis=1)
        
        dx_dt = lambda t,x,u: self.A.dot(x) + np.squeeze(self.B.dot(u), axis=1)
        
        result = integrate.solve_ivp(fun=dx_dt, args=(u,), y0=x0, t_span=t_span, method=self.algo, max_step=self.max_step, dense_output=True)

        return result.t, result.y.T
    
class feedbackController():
    def __init__(self, system):
        self.system = system
        self.model = system.model
        self.A = self.model['A']
        self.B = self.model['B']
        self.dimension = self.model['dimension']
        self.logX = Logger(log_dir='results')
        
    def step(self, x0, u, t_span):
        # u is the function of t and x (feedback law)
        dx_dt = lambda t,x: self.A.dot(x) + self.B.dot(u(t,x))
        result = integrate.solve_ivp(fun=dx_dt, y0=x0, t_span=t_span, method=self.system.algo, max_step=self.system.max_step, dense_output=True)
        return result.t, result.y.T        
        
    def lqr(self, Q=None, R=None):
        if np.all(Q==None):
            Q = np.eye(self.A.shape[0])
 
        if np.all(R==None):
            R = np.eye(self.B.shape[1])
            
        K,P,E = lqr(self.A, self.B, Q, R)
        
        return K, P
        
    def _isStable(self, A):
        eig = np.linalg.eig(A)[0].real
        return np.all(eig<0)
            
    def onPolicy(self, stop_thres=1e-3, viz=True):
        #online data collection
        self.viz = viz
        x_plot = [self.system.x0]       # list of array
        t_plot = [self.system.t_sim[0]]
        save_P = []
        save_K = [self.K0]
        K = save_K[-1]
        iter = 0
        
        while t_plot[-1] < self.system.t_sim[1]:
            u = lambda t,x: -K.dot(x) + self.explore_noise(t)
            
            theta = []
            xi = []
            # collect data_per_eval x data, iterate within an eval
            for i in range(self.num_data):
                x_sample = [x_plot[-1]]
                t_sample = [t_plot[-1]]
                t_collect = t_plot[-1]
                
                # collect online data, iterate within a sample time
                while t_plot[-1] < t_collect + self.data_eval:
                    t_temp, x_temp = self.step(x0=x_plot[-1], u=u, t_span=(t_plot[-1], t_plot[-1] + self.system.sample_time))
                    if self.viz:
                        self.logX.log('states_onPolicy', x_temp[-1], int(t_temp[-1]/self.system.sample_time))
                        
                    x_sample.append(x_temp[-1])
                    t_sample.append(t_temp[-1])
                    
                    x_plot.extend(x_temp[1:].tolist())
                    t_plot.extend(t_temp[1:].tolist())   
                    
                thetaRow, xiRow = self._rowGainOnPloicy(K, x_sample, t_sample)
                theta.append(thetaRow)           
                xi.append(xiRow)
            
            theta = np.array(theta)
            xi = np.array(xi)
            
            # check rank condition
            n,m = self.dimension
            if np.linalg.matrix_rank(theta) < m*n + n*(n+1)/2:
                print('not enough number of data, rank condition unsatisfied!')
                raise ValueError
            
            # solve P, K matrix
            PK = np.linalg.pinv(theta).dot(xi)
            P = PK[:n*n].reshape((n,n))
            if self.viz:
                self.logP.log('P_onPolicy', P, iter)
                self.logK.log('K_onPolicy', K, iter)
            save_P.append(P)
            # check stopping criteria
            try:
                err = np.linalg.norm(save_P[-1]-save_P[-2], ord=2)
                if err < stop_thres:
                    self.t_plot, self.x_plot = self._afterGainKopt(t_plot, x_plot, K, 'states_onPolicy')
                    break
            except: pass
            K = PK[n*n:].reshape((n,m)).T
            save_K.append(K)
            iter += 1
        return save_K[-1], save_P[-1]

    def _afterGainKopt(self, t_plot, x_plot, Kopt, section):
        # remove explore noise
        u = lambda t,x: -Kopt.dot(x)
        sample_time = self.system.sample_time
        start = t_plot[-1]
        stop = self.system.t_sim[1] 
        N = int((stop - start)/sample_time)
        for i in range(N):
            t_temp, x_temp = self.step(x0=x_plot[-1], u=u, t_span=(t_plot[-1], t_plot[-1] + sample_time))
            if self.viz:
                self.logX.log(section, x_temp[-1], int(t_temp[-1]/self.system.sample_time))
            x_plot.extend(x_temp[1:].tolist())
            t_plot.extend(t_temp[1:].tolist())  
                
        return t_plot, x_plot
                
    def _rowGainOnPloicy(self, K, x_sample, t_sample):
        xx = np.kron(x_sample[-1], x_sample[-1]) - np.kron(x_sample[0], x_sample[0])
        xeR = []
        xQx = []
        Qk = self.Q + K.T.dot(self.R.dot(K))
        # print(np.array(x_sample).shape)
        for i, xi in enumerate(x_sample):
            # print(type(xi))
            xi = np.array(xi)
            ei = self.explore_noise(t_sample[i])
            xeR.append(np.kron(xi, np.dot(ei,(self.R)).squeeze()))
            xQx.append(xi.dot(Qk.dot(xi)))
        
        xeR = -2*integrate.simpson(xeR, t_sample, axis=0)
        # xeR = -2*np.trapz(xeR, dx=sample_time, axis=0)
        _thetaRow = np.hstack((xx, xeR))
        _xiRow = -integrate.simpson(xQx, t_sample, axis=0)
        # _xiRow = -np.trapz(xQx, dx=sample_time, axis=0)
        return _thetaRow, _xiRow
        
    def setPolicyParam(self, K0=None, Q=None, R=None, data_eval=0.1, num_data=10, explore_noise=lambda t: 2*np.sin(100*t)):
        if np.all(Q==None):
            Q = np.eye(self.A.shape[0])
            
        if np.all(R==None):
            R = np.eye(self.B.shape[1])
        
        if np.all(K0==None):
            K0 = np.zeros(self.dimension).T
        if len(K0.shape)==1:
            K0 = np.expand_dims(K0, axis=0)
        # check stable of K0 
        if not self._isStable(self.A - self.B.dot(K0)):
            print('the inital K0 matrix is not stable, try re-initialize K0')
            raise ValueError
        
        self.K0 = K0
        self.Q = Q
        self.R = R
        self.data_eval = data_eval
        self.num_data = num_data
        self.explore_noise = explore_noise
        
        self.logK = Logger('results')
        self.logP = Logger('results')
            
    def offPolicy(self, stop_thres=1e-3, max_iter=30, viz=True):
        self.viz = viz
        self.stop_thres = stop_thres 
        self.max_iter = max_iter
        save_K = [self.K0]
        save_P = []

        x_plot = [self.system.x0]       # list of array
        t_plot = [self.system.t_sim[0]]
        
        u = lambda t,x: -self.K0.dot(x) + self.explore_noise(t)
        dxx = []
        Ixx = []
        Ixu = []
        for i in range(self.num_data):
            x_sample = [x_plot[-1]]
            t_sample = [t_plot[-1]]
            #collect data, iterate within data eval 
            t_collect = t_plot[-1]
            while t_plot[-1] < t_collect + self.data_eval:
                t_temp, x_temp = self.step(x0=x_plot[-1], u=u, t_span=(t_plot[-1], t_plot[-1] + self.system.sample_time))
                if self.visulize:
                    self.logX.log('states_offPolicy', x_temp[-1], int(t_temp[-1]/self.system.sample_time))
                
                x_sample.append(x_temp[-1])
                t_sample.append(t_temp[-1])
                
                x_plot.extend(x_temp[1:].tolist())
                t_plot.extend(t_temp[1:].tolist()) 
        
            dxx_ , Ixx_, Ixu_ = self._getRowOffPolicyMatrix(t_sample, x_sample)
            dxx.append(dxx_)
            Ixx.append(Ixx_)
            Ixu.append(Ixu_)
            
        # check rank condition
        test_matrix = np.hstack((Ixx, Ixu))
        n,m = self.dimension
        if np.linalg.matrix_rank(test_matrix) < m*n + n*(n+1)/2:
            print('not enough data, rank condition is not satisfied')
            raise ValueError
        
        # find optimal solution
        save_K, save_P = self._policyEval(dxx, Ixx, Ixu)
        self.Kopt = save_K[-1]
        self.t_plot, self.x_plot = self._afterGainKopt(t_plot, x_plot, self.Kopt, 'states_offPolicy')        
        # return optimal policy
        return save_K[-1], save_P[-1]
            
    def _policyEval(self, dxx, Ixx, Ixu):
        dxx = np.array(dxx)     # n_data x (n_state^2)
        Ixx = np.array(Ixx)     # n_data x (n_state^2)
        Ixu = np.array(Ixu)     # n_data x (n_state*n_input)
        
        save_K = [self.K0]
        save_P = []
        n,m = self.dimension
        K = save_K[-1]      # mxn
        for i in range(self.max_iter):
            temp = -2*Ixx.dot(np.kron(np.eye(n), K.T.dot(self.R)))  - 2*Ixu.dot(np.kron(np.eye(n), self.R))
            theta = np.hstack((dxx, temp))
            Qk = self.Q + K.T.dot(self.R.dot(K))
            xi = -Ixx.dot(Qk.ravel())
            
            PK = np.linalg.pinv(theta).dot(xi)
            P = PK[:n*n].reshape((n,n))
            if self.viz:
                self.logP.log('P_offPolicy', P, i)
                self.logK.log('K_offPolicy', K, i)
            save_P.append(P)
            # check stopping criteria
            try: 
                err = np.linalg.norm(save_P[-1] - save_P[-2], ord=2)
                if err < self.stop_thres:
                    break 
            except: pass
            K = PK[n*n:].reshape((n,m)).T
            save_K.append(K)
            
        return save_K, save_P   
        
    def _getRowOffPolicyMatrix(self, t_sample, x_sample):
        u = lambda t,x: -self.K0.dot(x) + self.explore_noise(t)

        dxx_ = np.kron(x_sample[-1], x_sample[-1]) - np.kron(x_sample[0], x_sample[0])
        xx = []
        xu = []
        for i, xi in enumerate(x_sample):
            xx.append(np.kron(xi, xi))
            ui = u(t_sample[i], xi)
            xu.append(np.kron(xi, ui))
            
        Ixx_ = integrate.simpson(xx, t_sample, axis=0)
        Ixu_ = integrate.simpson(xu, t_sample, axis=0)
        return dxx_, Ixx_, Ixu_
    
    
    
if __name__ == '__main__':
    ######################## Example 1: onPolicy method ####################
    # define system
    A = np.array([[0,1,0],[0,0,1],[-0.1, -0.5, -0.7]])
    B = np.array([0,0,1])
    system = linearSystem(A,B)
    # set simulation parameters
    max_step = 1e-3
    t_sim = (0,10)
    x0 = np.array([10, -10, 15])
    sample_time = 1e-3
    system.setSimulationParam(max_step=max_step, t_sim=t_sim, x0=x0, sample_time=sample_time)
    # pass system to controller
    controller = feedbackController(system)
    # set parameters for policy
    Q = np.eye(3); R = np.array([[1]]); K0 = np.zeros((1,3))
    explore_noise=lambda t: 2*np.sin(10*t)
    data_eval = 0.1; num_data = 10

    controller.setPolicyParam(K0=K0, Q=Q, R=R, data_eval=data_eval, num_data=num_data, explore_noise=explore_noise)
    # take simulation and get the results
    K, P = controller.onPolicy()
    print('On policy applied \n')
    print('Optimal policy K = \n', K, '\n Optimal value P = \n', P)



    # ######################## Example 1: offPolicy method ####################
    # # define system
    # A = np.array([[-0.4125, -0.0248, 0.0741, 0.0089, 0, 0], \
    #               [101.5873, -7.2615, 2.7608, 2.8068, 0, 0],
    #               [0.0704, 0.0085, -0.0741, -0.0089, 0, 0.02],
    #               [0.0878, 0.2672, 0, -0.3674, 0.0044, 0.3962],
    #               [-1.8414, 0.099, 0, 0, -0.0343, -0.033],
    #               [0, 0, 0, -359.0, 187.5364, -87.0316]])
    # B = np.array([[-0.0042, 0.0064], [-1.0360, 1.5849], [0.0042, 0], [0.1261, 0], [0, -0.0168], [0, 0]])
    # system = linearSystem(A,B)
    # # set simulation parameters
    # max_step = 1e-3
    # t_sim = (0, 5)
    # x0 = np.array([20, 5, 10, 2, -1, -2])
    # sample_time = 1e-3
    # system.setSimulationParam(max_step=max_step, t_sim=t_sim, x0=x0, sample_time=sample_time)
    # # pass system to controller
    # controller = feedbackController(system)
    # # set parameters for policy
    # Q = np.diag([100, 0, 0, 0, 0, 100]); R = np.eye(2); K0 = np.zeros((2,6))
    # data_eval = 0.01; num_data = 100
    # freq = ((np.random.rand(2,100)-0.5)*100).astype(int)
    # explore_noise = lambda t: np.sum(np.sin(freq*t), axis=1)

    # controller.setPolicyParam(K0=K0, Q=Q, R=R, data_eval=data_eval, num_data=num_data, explore_noise=explore_noise)
    # # take simulation and get the results
    # max_iter = 30
    # K, P = controllerffPolicy(max_iter=max_iter)
    # print('Off policy applied \n')
    # print('Optimal policy K = \n', K, '\n Optimal value P = \n', P)