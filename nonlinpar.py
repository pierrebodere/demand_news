import numpy as np

class NonLinParam:
    def __init__(self, alpha, theta_top_prop_gk, beta_g, sigma):
        """
        alpha: price coefficient
        theta_top_prop_gk: [G,K] array of preferences for topic proportion,
        first row is 0 (reference group)
        beta_g: [G,1] array of ideology by demographic group, 
        first element is 0 (reference group)
        sigma: unobserved heterogeneity of ideology
        """
        ## initialize objects
        self.alpha = alpha
        self.theta_top_prop_gk = theta_top_prop_gk
        self.beta_g = beta_g
        self.sigma = sigma
    
    @classmethod
    def from_array(cls, array, G_min_one, K):
        """
        array: a numpy array of non linear parameters
        G_min_one: 1 minus the number of clusters of demographics
        K: number of topics
        """
        alpha = array[0]
        theta_top_prop_gk = array[1 : (G_min_one +1) * K + 1].reshape((G_min_one+1, K))
        beta_g = np.concatenate(
            [np.array([0]), array[(G_min_one +1) * K + 1 : (G_min_one +1) * K + G_min_one + 1 ]]
        )
        sigma = array[-1]
        return cls(alpha, theta_top_prop_gk, beta_g, sigma)

    def to_array(self):
        """
        Flattens non lin parameters, removing the reference group (all 0)
        to be passed to optimization routine
        """
        return np.concatenate((
            np.array([self.alpha]), 
            self.theta_top_prop_gk.flatten(),
            self.beta_g[1:], 
            np.array([self.sigma])
        ))     
    
    
    