import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize

from utils import *
from v12_class_nonlinpar import *
from v12_class_market import *

class Economy:
    def __init__(self, markets, iv_gmm_weights_zz, Ntheta2, 
                 include_mm = True, sel_markets_mm = None,
                include_year_fe = False, include_newsp_fe = False):
        """
        markets: a collection of market objects
        iv_gmm_weights_zz: a matrix of weights 
        Ntheta2: the number of non linear parameters
        """
        self.markets = markets
        self.include_year_fe = include_year_fe
        self.include_newsp_fe = include_newsp_fe

        if sel_markets_mm is not None:
            self.markets_mm = [mkt for mkt in self.markets \
                               if mkt.market_index in sel_markets_mm]
        if sel_markets_mm is None:
            self.markets_mm = markets
            
        self.include_mm = include_mm
        ## Total number of products used in IV-GMM (obs) in markets
        self.N = sum([mkt.J_obs for mkt in markets])

        ## IV-GMM portion of the GMM weights (first Z, row, col)
        self.Z = self.markets[0].Z
        self.iv_gmm_weights_zz = iv_gmm_weights_zz

        ## Initialize GMM weights
        if self.include_mm:
            ## MM portion of GMM weights
            self.mm_gmm_weights = np.identity(int(Ntheta2 * \
                                                  sum([1 for mkt in self.markets_mm])))
        
            self.gmm_weights_hh = np.block([
                [self.iv_gmm_weights_zz, np.zeros((self.iv_gmm_weights_zz.shape[0], 
                                                   self.mm_gmm_weights.shape[1]))],
                [np.zeros((self.mm_gmm_weights.shape[0], self.iv_gmm_weights_zz.shape[1])),
                 self.mm_gmm_weights],
            ])
        else:
            self.gmm_weights_hh = self.iv_gmm_weights_zz

        self.G = self.markets[0].G
        self.K = self.markets[0].K
        
        self.Z_mm_zipshare = sum([mkt.J_obs * mkt.C for mkt in markets])
        self.optim_iter = 0


    
    def get_allmkt_delta(self, nonLinPar):
        """
        nonLinPar: a NonLinearParam object

        Returns
            [N,1] array of concatenated delta from BLP in each market
        Note
            Parallelize this operation over markets (does not go faster)
        """

        delta_n = np.concatenate([
            mkt.get_delta(nonLinPar).flatten()[np.argwhere(mkt.ind_obs_text_j==1).flatten()] \
            for mkt in self.markets])
        
        np.concatenate([
            mkt.get_delta(nonLinPar).flatten()[np.argwhere(mkt.ind_obs_text_j==1).flatten()] \
            for mkt in self.markets_mm])

        return delta_n


    def get_linear_char(self):
        """
        Returns
            [N, Klinear + 1 ] array of linear product characteristics, 
            including all the relevant fixed effects
        """
        
        market_lin_char = np.concatenate(
            [mkt.exog_lin_char_jk1[np.argwhere(mkt.ind_obs_text_j==1).flatten(),:] \
                for mkt in self.markets])
        
        if self.include_year_fe:
            market_years = np.concatenate(
                [np.repeat(mkt.market_year, mkt.J_obs) for mkt in self.markets]
            )
            year_dummies =  np.zeros((market_years.size, len(np.unique(market_years))))
            year_dummies[np.arange(market_years.size), market_years-min(market_years)] = 1
            market_lin_char= np.column_stack((market_lin_char, year_dummies[:,1:]))
            
        if self.include_newsp_fe:
            market_newsp = np.concatenate(
                [mkt.market_newsp[np.argwhere(mkt.ind_obs_text_j==1).flatten()] for mkt in self.markets]
            )
            unique_markets_newsp = dict(zip(np.unique(market_newsp), 
                                           range(len(np.unique(market_newsp)))))
            market_newsp = np.array([unique_markets_newsp[i] for i in market_newsp]) # replace newsp by their index
            newsp_dummies =  np.zeros((market_newsp.size, len(np.unique(market_newsp))))
            newsp_dummies[np.arange(market_newsp.size), market_newsp] = 1
            market_lin_char = np.column_stack((market_lin_char, newsp_dummies[:,1:]))
            
        return market_lin_char

    def get_inst(self):
        """
        Returns
            [N, Z] array of instruments over all products
        """
        X_nk = self.get_linear_char()
        inst_jz = np.column_stack((
            X_nk, 
            np.concatenate([mkt.excluded_inst_jz[np.argwhere(mkt.ind_obs_text_j==1).flatten(),:] \
                               for mkt in self.markets])
        ))
        
        return inst_jz

    def get_beta_k1(self, nonLinPar, delta_n):
        """
        nonLinPar: a NonLinearParam object

        Returns
            [N,1] array of residuals resulting from linear IV-GMM of delta 
            on linear characteristic instrumented by Z
        """
        ## get matrices of exogenous char, instruments
        X_nk = self.get_linear_char()
        Z_nz = self.get_inst()

        ## get LHS variable delta
        #delta_n = self.get_allmkt_delta(nonLinPar)

        ## IV-GMM (Conlon, Gortmaker)
        tilde_Y_z = (1/self.N) * Z_nz.T @ delta_n
        tilde_X_zk = (1/self.N) *  Z_nz.T @ X_nk
        hat_beta_k = np.linalg.inv(tilde_X_zk.T @ self.iv_gmm_weights_zz \
            @ tilde_X_zk) @ tilde_X_zk.T @ self.iv_gmm_weights_zz @ tilde_Y_z
        
        if self.optim_iter % 10 == 0:
            print("linear parameters at iter " + str(self.optim_iter), flush=True)
            print(hat_beta_k, flush=True)
        
        return(hat_beta_k)
    


    def get_xi(self, nonLinPar):
        """
        nonLinPar: a NonLinearParam object

        Returns
            [N,1] array of residuals resulting from linear IV-GMM of delta 
            on linear characteristic instrumented by Z
        """
        ## get matrices of exogenous char, instruments
        X_nk = self.get_linear_char()

        ## get LHS variable delta
        delta_n = self.get_allmkt_delta(nonLinPar)

        ## IV-GMM (Conlon, Gortmaker)
        hat_beta_k = self.get_beta_k1(nonLinPar, delta_n)
        
        xi_n = delta_n - X_nk @ hat_beta_k
        return(xi_n)


    def gmm_moments(self, nonLinPar):
        """
        nonLinPar: a NonLinearParam object

        Returns
            [H,1] array of GMM non linear moment conditions from interacting 
            xi with instruments
        """
        ## get instruments
        Z_nz = self.get_inst()

        ## get residuals xi
        xi_n = self.get_xi(nonLinPar)

        ## form IV demand moments
        g_z = (1/self.N) * Z_nz.T @ xi_n
        
        
        ## add other moments
        if self.include_mm:
            g_mm_zipshare = np.concatenate([
                mkt.g_mm_zipshare(nonLinPar, mkt.res_delta_j) for mkt in self.markets_mm
            ])

            g_z = np.concatenate((g_z, g_mm_zipshare))
        
        ## return
        return g_z
    
        

    def gmm_obj(self, array, quiet= True):
        """
        array: an array concatenating all nonLinPar component

        Returns
            Scalar GMM objective function evaluated at array that we want to minimize
        """
        t_obj = time.time()
        nonLinPar = NonLinParam.from_array(array, self.G-1, self.K)
        
        ## get gmm moments
        g_h = self.gmm_moments(nonLinPar)

        ## form gmm objective and return
        q = g_h.T @ self.gmm_weights_hh @ g_h
        
        if not quiet:
            print("gmm obj: " + str(q), flush = True)
            print("     Time to compute obj: " + str(time.time()-t_obj),flush = True)
        if self.optim_iter % 10 == 0:
            print("non linear parameters at iter " + str(self.optim_iter), flush=True)
            print(array, flush=True)
        self.optim_iter += 1
            
        return q
    
    def ddelta_dtheta_nk2(self, nonLinPar):
        """
        nonLinPar: a NonLinearParam object

        Returns
            [Z, K2] array, Jacobian of the GMM objective
        """
        dxi_ddelta_nk2 = np.concatenate([
            mkt.ddelta_dtheta2(nonLinPar, mkt.res_delta_j)[np.argwhere(mkt.ind_obs_text_j==1).flatten(),:] \
            for mkt in self.markets])
        return dxi_ddelta_nk2

    def get_jac_gmm_obj(self, nonLinPar):
        """
        nonLinPar: a NonLinearParam object

        Returns
            [Z, K2] array, Jacobian of the GMM objective
        """
        dxi_ddelta_nk2 = np.concatenate([
            mkt.ddelta_dtheta2(nonLinPar, mkt.res_delta_j)[np.argwhere(mkt.ind_obs_text_j==1).flatten(),:] \
            for mkt in self.markets])

        inst_nz = self.get_inst()

        return (1/self.N) * inst_nz.T @ dxi_ddelta_nk2      


    def get_gradient_gmm_obj(self, array, quiet= True):
        """
        array: an array concatenating all nonLinPar component

        Returns
            [Z, ] array, gradient the GMM objective
        """
        t0 = time.time()
        nonLinPar = NonLinParam.from_array(array, self.G-1, self.K)
        
        ## get moments 
        g_h1 = self.gmm_moments(nonLinPar)
        
        ## Jacobians
        G_zk2 = self.get_jac_gmm_obj(nonLinPar)
        
        if self.include_mm:
            G_mm_k2 = np.concatenate([mkt.dg_mm_zipshare_dtheta2_alt(nonLinPar) for mkt in self.markets_mm])
            G_zk2 = np.concatenate((G_zk2, G_mm_k2))

        ## Compute gradient
        grad_k2 = 2 * G_zk2.T @ self.gmm_weights_hh @ g_h1[:, np.newaxis]
        
        if not quiet:
            print("      Time to compute grad: " + str(time.time()-t0), flush = True)
        return grad_k2.flatten()        

    def opt_gmm_ob(self, startNonLinPar, bounds, quiet = True):
        """
        Function optimizing the gmm objective function within constraints

        startNonLinPar: an initial non Linear Param Object
        
        """
        startarray = startNonLinPar.to_array()

        x = minimize(
            self.gmm_obj, 
            startarray, 
            args= (quiet),
            method = "L-BFGS-B", 
            bounds=bounds, 
            jac = self.get_gradient_gmm_obj,
            options = {"gtol":10e-4}
        )
        return x
        
        
    ## Variance
    def inv_weigths_second_step(self, nonLinPar):

        ## S_iv_zz
        Z_nz = self.get_inst()
        xi_n = self.get_xi(nonLinPar)
        S_zz = (1/self.N) * (Z_nz * xi_n[:, np.newaxis]).T @ (Z_nz * xi_n[:, np.newaxis])
        
        ## Add variance of mm
        if self.include_mm:
            for mkt in self.markets_mm:
                S_zz = block_diag(
                    S_zz, 
                    mkt.g_mm_variance_nn(nonLinPar)
                )
        return S_zz
    
    def std_errors_estimates(self, nonLinPar):
        ## Jacobians
        G_hk2 = self.get_jac_gmm_obj(nonLinPar)
        if self.include_mm:
            G_mm_k2 = np.concatenate([mkt.dg_mm_zipshare_dtheta2_alt(nonLinPar) for mkt in self.markets_mm])
            G_hk2 = np.concatenate((G_hk2, G_mm_k2))
        
        ## Compute std errors
        bread_kk = np.linalg.pinv(G_hk2.T @ self.gmm_weights_hh @ G_hk2)
        S_zz = self.inv_weigths_second_step(nonLinPar)
        V_kk = bread_kk @ G_hk2.T @ self.gmm_weights_hh @ S_zz @ self.gmm_weights_hh @ G_hk2 @ bread_kk
        
        ## Return standard errors
        se_k = np.sqrt(np.diag(V_kk/self.N))
        
        return se_k
