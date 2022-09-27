import numpy as np
from statsmodels.tools.numdiff import approx_fprime_cs
from joblib import Parallel, delayed
from utils import *
from nonlinpar import *
import time


class Market:
    def __init__(
        self, market_index, market_year, market_newsp, 
        weights_cg, weights_c1,
        obs_s_j,
        obs_s_cj, init_delta_j, p_j,
        lambda_jk, v_j, 
        exog_lin_char_jk1, excluded_inst_jz,
        ind_obs_text_j,
        true_delta_j = None
    ):
        """ 
            weights_cg: [C, G] array of share of each group in in each zipcode of the market
            weights_c1: [C, 1] array of share of population in in each zipcode of the market
            obs_s_j: [J,1] array of newspapers share in the market
            obs_s_cj: [C, J] array of newspapers share broken down by zipcode
            init_delta_j: [J,1] array of initial guess for delta
            p_j: [J,1] array of newspapers price
            lambda_jk: [J,K] array of newspaper topic proportion
            v_j: [J,1] array of newspaper ideology
            exog_lin_char_jk1: [J, K1] array of characteristics entering delta; includes intercept,
            price, topic proportions (for reference cluster) and potentially FE
            excluded_inst_jz: [J, Z] array of instruments, not comprising linear char
            ind_obs_text_j: [J,1] array of indicator for whether text characteristics are observed
        """
        self.market_index = market_index
        self.market_year = market_year
        self.market_newsp = market_newsp
        
        self.G = weights_cg.shape[1]
        self.C = weights_cg.shape[0]
        self.Q = 7
        
        ## initialize weights
        nodes_i, quad_weights_i = np.polynomial.hermite.hermgauss(self.Q)
        ### repeat weights to get a vector of size C * G * Q
        quad_weights_i1 = np.concatenate([quad_weights_i] * self.C * self.G)
        g_weights_i1 = np.repeat(weights_cg, self.Q)
        c_weights_i1 = np.repeat(weights_c1, self.G * self.Q)
        
        self.weights_c1= weights_c1
        self.weights_i = c_weights_i1 * g_weights_i1 * quad_weights_i1
        self.weights_agg_zip_i = g_weights_i1 * quad_weights_i1
        self.nodes_i = np.concatenate([nodes_i] * self.C * self.G)
        
        ## Define selectors 
        self.G_ig = np.zeros((self.Q * self.G * self.C, self.G))
        for g in range(self.G):
            self.G_ig[g * self.Q * self.C: (g+1) * self.Q * self.C, g] =1
            
        self.C_ic = np.zeros((self.Q * self.G * self.C, self.C))
        for c in range(self.C):
            self.C_ic[c * self.G * self.Q : (c+1) * self.G * self.Q  , c] = 1
            
        ## initialize objects
        self.obs_s_j = obs_s_j
        self.obs_s_cj = obs_s_cj
        self.init_delta_j = init_delta_j
        self.res_delta_j = None
        self.p_j = p_j
        self.lambda_jk = lambda_jk
        self.v_j = v_j
        self.exog_lin_char_jk1 = exog_lin_char_jk1
        self.excluded_inst_jz = excluded_inst_jz
        self.inst_jz = np.column_stack((self.exog_lin_char_jk1 , self.excluded_inst_jz))
        self.ind_obs_text_j = ind_obs_text_j

        ## initialize object dimensions
        self.I = self.G_ig.shape[0]
        self.J = p_j.shape[0]
        self.J_obs = np.sum(ind_obs_text_j)
        self.K = lambda_jk.shape[1]
        self.Z = self.inst_jz.shape[1]

        self.res_delta_j = None
        self.true_delta_j = true_delta_j ## for test and simulations

    def compute_mu(self, nonLinPar):
        """
        nonLinPar: a NonLinearParam object 

        Returns:
            [I,J] array of non linear portion of utility mu_ij
        """
        ## utils from price sensitivity
        alpha_g = np.repeat(np.exp(np.array([nonLinPar.alpha])), self.G)
        mu_price_ij = - (self.G_ig @ alpha_g)[:, np.newaxis] * self.p_j[np.newaxis, :]
        
        ## utils from topic prop
        mu_top_prop_ij = self.G_ig @ np.exp(nonLinPar.theta_top_prop_gk) @ self.lambda_jk.transpose()
        
        ## utils from ideology
        beta_v_i = self.G_ig @ nonLinPar.beta_g + nonLinPar.sigma * self.nodes_i
        mu_ideo_ij = beta_v_i[:, np.newaxis] @ self.v_j[np.newaxis,:]

        return mu_price_ij + mu_top_prop_ij + mu_ideo_ij

    def compute_utils(self, nonLinPar, delta_j):
        """
        nonLinPar: a NonLinearParam object
        delta_j: [J,1] array of linear portion of utility 

        Returns:
            [I,J] array of utility u_ij
        """
        mu_ij = self.compute_mu(nonLinPar)
        
        ## replace mu by 0 when we do not observe text
        mu_ij = mu_ij * self.ind_obs_text_j
        
        return np.tile(delta_j, self.I).reshape((self.I, self.J)) + mu_ij
        
    def compute_node_shares(self, nonLinPar, delta_j):
        """
        nonLinPar: a NonLinearParam object
        delta_j: [J,1] array of linear portion of utility 

        Returns:
            [I,J] array of agent specific market shares s_ij
        """
        u_ij = self.compute_utils(nonLinPar, delta_j)
        denom_i = 1 + np.exp(logsumexp(u_ij, axis = 1)) # robust to overflow
        denom_ij = np.tile(denom_i.reshape(self.I, 1), self.J).reshape((self.I, self.J))
        return np.exp(u_ij) / denom_ij

    def compute_shares(self, nonLinPar, delta_j):
        """
        nonLinPar: a NonLinearParam object
        delta_j: [J,1] array of linear portion of utility 

        Returns:
            [J,1] array of utility aggregate market shares s_j
        """
        s_ij = self.compute_node_shares(nonLinPar, delta_j)
        s_j = self.weights_i.reshape((1,self.I)) @ s_ij
        return s_j
 
    def BLP_fixed_point(self, delta_step_j, nonLinPar ):
        """
        nonLinPar: a NonLinearParam object
        delta_step_j: [J,1] array of linear portion of utility 

        Returns:
            [J,1] array of next step linear portion of utility
        """
        s_j = self.compute_shares(nonLinPar, delta_step_j)
        res_1j = delta_step_j + np.log(self.obs_s_j) - np.log(s_j)
        res = res_1j.T[:,0]
        return res
        
    def get_delta(self, nonLinPar, setResDelta = True):
        """
        nonLinPar: a NonLinearParam object

        Returns:
            [J,] array, the fixed point of the BLP contraction mapping
        """
        gap = 1
        prev_delta_j = self.init_delta_j
        while gap > 1e-12:
            delta_j = self.BLP_fixed_point(prev_delta_j, nonLinPar)
            gap = np.max(np.abs(delta_j - prev_delta_j))
            prev_delta_j = delta_j
        if setResDelta:
            self.res_delta_j = delta_j
        return delta_j

    def ds_ddelta(self, nonLinPar, delta_j):
        """
        nonLinPar: a NonLinearParam object
        delta_j: [J,1] array of linear portion of utility
        
        Returns:
            [J,J] array, the derivative of the shares wrt to delta
        """
        s_ij = self.compute_node_shares(nonLinPar, delta_j)
        ws_ij = self.weights_i[:, np.newaxis] * s_ij
        off_diag_jj = - ws_ij.T @ s_ij
        diag_jj = np.diag(np.sum(self.weights_i[:, np.newaxis] * s_ij, axis = 0))
        return off_diag_jj + diag_jj
    
    
    def dmu_dalpha(self, nonLinPar):
        """ 
        Returns:
            [I, J] array, the derivative of mu_ij wrt alpha0_g    
        """
        alpha_g = np.repeat(np.exp(np.array([nonLinPar.alpha])), self.G)
        temp = - (self.G_ig @ alpha_g)[:, np.newaxis] * self.p_j[np.newaxis, :]
        return temp

    def dmu_dtheta_prop_gk(self, nonLinPar):
        """ 
        Returns:
            [I, J, G-1 * K] array, the derivative of mu_ij wrt theta_top_prop_gk    
        """
        temp = self.G_ig[:, np.newaxis, :, np.newaxis] *\
            np.exp(nonLinPar.theta_top_prop_gk)[np.newaxis, np.newaxis, :, :] *\
            self.lambda_jk[np.newaxis, :, np.newaxis, :]
        temp = temp.reshape((self.I, self.J, self.G * self.K))
        return temp
    
    def dmu_dbeta_g(self):
        """ 
        nonLinPar: a NonLinearParam object

        Returns:
            [I, J, G-1] array, the derivative of mu_ij wrt beta    
        """
        deriv = self.G_ig[:, np.newaxis, :] * self.v_j[np.newaxis, :, np.newaxis]
        return deriv[:, :, 1:]
    
    def dmu_dsigma(self):
        """ 
        nonLinPar: a NonLinearParam object

        Returns:
            [I, J] array, the derivative of mu_ij wrt sigma    
        """
        return self.nodes_i[:, np.newaxis] * self.v_j[np.newaxis, :]
    

    def dmu_dtheta2(self, nonLinPar):
        """ nonLinPar: a NonLinearParam object
        
        Returns:
            [I, J, Ntheta2] array, the derivative of mu_ij wrt theta2
        """
        return np.concatenate((
            self.dmu_dalpha(nonLinPar)[:, :, np.newaxis],
            self.dmu_dtheta_prop_gk(nonLinPar),
            self.dmu_dbeta_g(),
            self.dmu_dsigma()[:, :, np.newaxis]
        ), axis = 2)

    def ds_dtheta2(self, nonLinPar, delta_j):
        """ nonLinPar: a NonLinearParam object
            delta_j: [J,1] array of linear portion of utility 
            
        Returns:
            [J, Ntheta2] array, the derivative of s_j wrt theta2
        """
        dmu_dtheta_ijn = self.dmu_dtheta2(nonLinPar)
        s_ij = self.compute_node_shares(nonLinPar, delta_j)
        s_ij = s_ij * self.ind_obs_text_j
        dmudtheta_s_in = np.sum(dmu_dtheta_ijn * s_ij[:, :, np.newaxis], axis = 1)
        ws_ijn = self.weights_i[:, np.newaxis, np.newaxis] * s_ij[:, :, np.newaxis]
        return np.sum(ws_ijn * (dmu_dtheta_ijn - dmudtheta_s_in[:, np.newaxis, :]), axis= 0)
    
    def dnodeshare_ddelta_ijj(self, nonLinPar, delta_j):
        s_ij = self.compute_node_shares(nonLinPar, delta_j)
        ds_ddelta_ijj = np.array([-s_ij[:,k][:,np.newaxis] * s_ij \
                                  for k in range(s_ij.shape[1])]).transpose((1,2,0))
        for j in range(s_ij.shape[1]):
            ds_ddelta_ijj[:,j,j] += s_ij[:,j]
        return ds_ddelta_ijj
    

    def ddelta_dtheta2(self, nonLinPar, delta_j):
        """ 
        nonLinPar: a NonLinearParam object

        Returns:
        [J, Ntheta2] array, the derivative of delta_j wrt theta2

        Note: 
        This code computes delta j every time. 
        Could be a good idea to only compute if true delta j has not been 
        computed yet, to speed up the code
        """
        where_obs = np.argwhere(self.ind_obs_text_j==1).flatten()
        ds_ddelta_jj = self.ds_ddelta(nonLinPar, delta_j)
        ds_ddelta_obs_jj = ds_ddelta_jj[where_obs][:,where_obs]

        ds_dtheta2_jn = self.ds_dtheta2(nonLinPar, delta_j)
        ds_dtheta2_obs_jn = ds_dtheta2_jn[where_obs]
        
        ddelta_dtheta_all_jn = np.zeros((self.J,ds_dtheta2_obs_jn.shape[1]))

        ddelta_dtheta_all_jn[where_obs] = \
            - np.linalg.inv(ds_ddelta_obs_jj) @ ds_dtheta2_obs_jn
        return ddelta_dtheta_all_jn
    
    
    
    ## Micromoments
    def compute_share_zip_code_jc(self, nonLinPar, delta_j):
        """
        Returns:
            s_jc: [J, C] array of zip code level circulation share for each newspaper
        """
        s_ij = self.compute_node_shares(nonLinPar, delta_j)
        s_jc = (self.weights_agg_zip_i[:, np.newaxis] * s_ij).T @ self.C_ic
        return s_jc
    
    
    def ds_jc_dtheta2(self, nonLinPar, delta_j):
        """
        Returns:
          [J, C, Ntheta2] array of the derivative of s_jc wrt theta2, 
          which is then used to compute the microments
        """
        dmu_dtheta_ijn = self.dmu_dtheta2(nonLinPar)
        s_ij = self.compute_node_shares(nonLinPar, delta_j)
        dmudtheta_s_in = np.sum(dmu_dtheta_ijn * s_ij[:, :, np.newaxis], axis = 1)
        ws_ijn = self.weights_agg_zip_i[:, np.newaxis, np.newaxis] * s_ij[:, :, np.newaxis]
        dsij_dtheta_ijn = ws_ijn * (dmu_dtheta_ijn - dmudtheta_s_in[:, np.newaxis, :])
        dsjc_d_theta_jnc = np.transpose(dsij_dtheta_ijn, (1,2,0)) @ self.C_ic    
        return np.transpose(dsjc_d_theta_jnc, (0,2,1))
    
    def ds_jc_dtheta2_alt(self, nonLinPar, delta_j):
        """
        Returns:
          [J, C, Ntheta2] array of the derivative of s_jc wrt theta2, 
          which is then used to compute the microments
        """
        dmu_dtheta_ijn = self.dmu_dtheta2(nonLinPar)
        s_ij = self.compute_node_shares(nonLinPar, delta_j)
        dmudtheta_s_in = np.sum(dmu_dtheta_ijn * s_ij[:, :, np.newaxis], axis = 1)
        
        s_dmu_dtheta_ijn = s_ij[:, :, np.newaxis] * (dmu_dtheta_ijn - dmudtheta_s_in[:, np.newaxis, :])
        
        ## aggregate over zips
        wC_ic = self.weights_agg_zip_i[:, np.newaxis] * self.C_ic
        res_j = []
        for j in range(s_dmu_dtheta_ijn.shape[1]):
            res_n = []
            for n in range(s_dmu_dtheta_ijn.shape[2]):
                res_n.append(s_dmu_dtheta_ijn[:,j,n] @ (wC_ic))
            res_j.append(np.array(res_n))
        dsjc_dtheta_jnc = np.array(res_j)
        dsjc_dtheta_jcn = dsjc_dtheta_jnc.transpose((0, 2, 1))
        
        return dsjc_dtheta_jcn
    
    def ds_jc_dtheta2_ddelta(self, nonLinPar, delta_j):
        """
        """
        dmu_dtheta_ijn = self.dmu_dtheta2(nonLinPar)
        s_ij = self.compute_node_shares(nonLinPar, delta_j)
        dnode_ddelta_ijk = self.dnodeshare_ddelta_ijj(nonLinPar, delta_j)
        
        ## get sum_k dmu s and sum_k dmu dsdelta
        dmudtheta_s_in = np.sum(dmu_dtheta_ijn * s_ij[:, :, np.newaxis], axis = 1)
        dmudtheta_dsdelta_ikn = np.sum(dmu_dtheta_ijn[:,:,np.newaxis,:] *\
                                       dnode_ddelta_ijk[:,:,:,np.newaxis], axis = 1)
        
        ## get w s and w dsdelta
        ws_ijn = self.weights_agg_zip_i[:, np.newaxis, np.newaxis] * s_ij[:, :, np.newaxis]
        wdsdelta_ijkn = self.weights_agg_zip_i[:, np.newaxis, np.newaxis, np.newaxis] * \
            dnode_ddelta_ijk[:, :, :, np.newaxis]
        
        ## add the 2 terms of the cross product
        dsij_dtheta_ddelta_ijkn = wdsdelta_ijkn * \
            (dmu_dtheta_ijn[:,:,np.newaxis,:] - dmudtheta_s_in[:, np.newaxis, np.newaxis, :]) -\
            ws_ijn[:,:,np.newaxis,:] * dmudtheta_dsdelta_ikn[:,np.newaxis, :,:]
        
        
        dsjc_d_theta_ddelta_jknc = np.transpose(dsij_dtheta_ddelta_ijkn, (1,2,3,0)) @ self.C_ic 

        dsjc_d_theta_ddelta_jcnk = dsjc_d_theta_ddelta_jknc.transpose((0,3,2,1))
        return dsjc_d_theta_ddelta_jcnk
    
        
    def g_mm_zipshare(self, nonLinPar, delta_j):
        """
        Returns:
           N*Jobs*C micromoments (from Jimenez-Hernandez, Seira)
        """
    
        ## observed shares
        where_obs = np.argwhere(self.ind_obs_text_j==1).flatten()
        obs_s_jc = self.obs_s_cj.T
        obs_s_jcn = np.dstack((obs_s_jc,) * nonLinPar.to_array().shape[0])
        obs_s_jcn_obs = obs_s_jcn[where_obs, :, :]
        
        ## model share
        s_jc = self.compute_share_zip_code_jc(nonLinPar, delta_j)
        s_jcn = np.dstack((s_jc,) * nonLinPar.to_array().shape[0])
        s_jcn_obs = s_jcn[where_obs, :, :]

        ## derivative of model shares
        ds_jc_dtheta2_jcn = self.ds_jc_dtheta2(nonLinPar, delta_j)
        ds_jc_dtheta2_jcn_obs = ds_jc_dtheta2_jcn[where_obs, :, :]
        
        ## average over all zips
        #weights_c= self.weights_c1[:, 0]
        weights_c= self.weights_c1
        wg_mm_jcn = weights_c[np.newaxis, :, np.newaxis] * ds_jc_dtheta2_jcn_obs * \
        (s_jcn_obs - obs_s_jcn_obs)
        wg_mm_jn = wg_mm_jcn.sum(axis = 1)
        
        ## average over j
        wg_mm_n = wg_mm_jn.mean(axis = 0)
        
        return wg_mm_n.flatten()
        
    ## Derivative micro moments
    def dg_mm_zipshare_dtheta2_part(self, nonLinPar, delta_j):
        ## define a local function to diff
        def loc_g(arg):
            nonLinParFromArray = NonLinParam.from_array(arg, self.G-1, self.K) 
            g_mm = self.g_mm_zipshare(nonLinParFromArray, delta_j)
            return g_mm
        ## differentiate numerically
        diff_g_mm_n = approx_fprime_cs(nonLinPar.to_array(), loc_g)
        return diff_g_mm_n

    def dg_mm_zipshare_ddelta_part(self, nonLinPar, delta_j):
        where_obs = np.argwhere(self.ind_obs_text_j==1).flatten()
        s_ij = self.compute_node_shares(nonLinPar, delta_j)
        ds_jc_dtheta2_ddelta_jcnk = self.ds_jc_dtheta2_ddelta(nonLinPar, delta_j)
        ds_jc_dtheta2_jcn = self.ds_jc_dtheta2(nonLinPar, delta_j)
        d_sij_ddelta_ijk = self.dnodeshare_ddelta_ijj(nonLinPar, delta_j)
        weights_agg_zip_i = self.weights_agg_zip_i 
        obs_s_jc = self.obs_s_cj.T
        s_jc = (self.weights_agg_zip_i[:, np.newaxis] * s_ij).T @ self.C_ic
        
        ## aggregate over zips
        wC_ic = weights_agg_zip_i[:, np.newaxis] * self.C_ic
        res_j = []
        for j in range(d_sij_ddelta_ijk.shape[1]):
            res_k = []
            for k in range(d_sij_ddelta_ijk.shape[2]):
                res_k.append(d_sij_ddelta_ijk[:,j,k] @ (wC_ic))
            res_j.append(np.array(res_k))
        dsjc_ddelta_jkc = np.array(res_j)
        dsjc_ddelta_jck = dsjc_ddelta_jkc.transpose((0, 2, 1))
        
        ## sum two parts of derivative
        res_jcnk = ds_jc_dtheta2_ddelta_jcnk * \
            (s_jc[:, :, np.newaxis, np.newaxis] - obs_s_jc[:, :, np.newaxis, np.newaxis]) +\
            ds_jc_dtheta2_jcn[:, :, :, np.newaxis] * dsjc_ddelta_jck[:, :, np.newaxis, :]
        
        ## aggregate over zip 
        res_jnk = np.sum(self.weights_c1[np.newaxis, :, np.newaxis, np.newaxis] * res_jcnk, axis = 1)
        res_jobsnk = res_jnk[where_obs, :, :]
        
        ## aggregate over J
        res_nk = res_jobsnk.mean(axis = 0)
        #res_jn_k = np.array([res_jobsnk[:, :, k].flatten() for k in range(res_jobsnk.shape[2])]).T
        return res_nk
    
    
    
    def dg_mm_zipshare_dtheta2_alt(self, nonLinPar):
        where_obs = np.argwhere(self.ind_obs_text_j==1).flatten()
        if self.res_delta_j is None:
            self.get_delta(nonLinPar)
        res_delta_j = self.res_delta_j
        dg_mm_zipshare_dtheta2_part_mmn = self.dg_mm_zipshare_dtheta2_part(nonLinPar, res_delta_j)
        dg_mm_zipshare_ddelta_part_mmj = self.dg_mm_zipshare_ddelta_part(nonLinPar, res_delta_j)
        ddelta_dtheta_2_jn = self.ddelta_dtheta2(nonLinPar, res_delta_j)

        diff_g_mm_n = dg_mm_zipshare_ddelta_part_mmj @ ddelta_dtheta_2_jn + dg_mm_zipshare_dtheta2_part_mmn
        return diff_g_mm_n
    
    
    
    ## Variance micromoments
    def g_mm_variance_nn(self, nonLinPar):
        
        ## observed shares
        where_obs = np.argwhere(self.ind_obs_text_j==1).flatten()
        obs_s_jc = self.obs_s_cj.T
        obs_s_jcn = np.dstack((obs_s_jc,) * nonLinPar.to_array().shape[0])
        obs_s_jcn_obs = obs_s_jcn[where_obs, :, :]

        
        ## model share
        s_jc = self.compute_share_zip_code_jc(nonLinPar, self.res_delta_j)
        s_jcn = np.dstack((s_jc,) * nonLinPar.to_array().shape[0])
        s_jcn_obs = s_jcn[where_obs, :, :]

        ## derivative of model shares
        ds_jc_dtheta2_jcn = self.ds_jc_dtheta2(nonLinPar, self.res_delta_j)
        ds_jc_dtheta2_jcn_obs = ds_jc_dtheta2_jcn[where_obs, :, :]

        ## get covariance
        #weights_c = self.weights_c1[:, 0]
        g_jcn = ds_jc_dtheta2_jcn_obs * (s_jcn_obs - obs_s_jcn_obs)
        g_nc = g_jcn.mean(axis = 0).T
        #g_flat_jn_c = np.concatenate(list(g_jcn.transpose(0,2,1)))
        cov_n_n = np.cov(g_nc, aweights = self.weights_c1) 
        
        return cov_n_n
        
        
        