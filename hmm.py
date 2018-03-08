import numpy as np
import scipy.stats as ss
from mfcc import ret_mfcc
import scipy
from silence_detect import find_speech
def normalize(x):
    return (x + (x == 0)) / np.sum(x)

def stochasticize(x):
    return (x + (x == 0)) / np.sum(x, axis=1)

class HMM:
    #param init goes here
    def __init__(self, name, nstates):
        #self.mfcc = mfcc # do i want to store this?
        self.num_states = nstates # N
        self.init_state = np.random.RandomState(0)
        self.prior = normalize(self.init_state.rand(self.num_states, 1)) #pi
        self.A = stochasticize(self.init_state.rand(self.num_states, self.num_states)) #transition matrix
        self.name = name
        self.num_frames = None
        self.n_dims = None
        self.mu = None
        self.covs = None

    #assume inputs are array of log nums
    def _log_mult(self, X):
        ret = 0
        for x in X:
            ret += x
        return ret

    #assume inputs are array of nums log
    def _log_add(self, X):
        ret = X[0]
        for x in range(1,len(X)):
            a = ret
            b = x
            l0 = -1 * 10**30
            if a > b:
                temp = a
                a = b
                b = temp
            d = a - b
            if d < -np.log(-l0):
                ret = b
            else:
                ret = b + np.log(1 + np.exp(a - b))
        return ret

    def _trellis_init(self,mfcc):
        #N x T matrix
        #init a trellis var
        self.num_frames = len(mfcc[0]) # T
        self.n_dims = len(mfcc) # should be 26
        subset = self.init_state.choice(np.arange(self.n_dims), size=self.num_states, replace=False)
        self.mu = mfcc[:, subset]
        self.covs = np.zeros((self.n_dims, self.n_dims, self.num_states))
        self.covs += np.diag(np.diag(np.cov(mfcc)))[:, :, None]
        trellis = np.zeros((self.num_states, self.num_frames))
        for n in range(0, self.num_states):
            trellis[n, :] = ss.multivariate_normal.pdf(mfcc.T, mean=self.mu[:, n].T, cov=self.covs[:, :, n].T)
        return trellis

    def _alpha_recursion(self, mfcc, trellis):
        log_likelihood = 0
        T = self.num_frames
        alpha = np.zeros(trellis.shape)
        for t in range(0,T):
            if t == 0:
                alpha[:, t] = trellis[:, t] * self.prior.ravel()
            else:
                alpha[:, t] = trellis[:, t] * np.dot(self.A.T, alpha[:, t - 1])
            alpha_sum = np.sum(alpha[:, t])
            alpha[:, t] /= alpha_sum #so we get prob < 1
            log_likelihood = log_likelihood + np.log(alpha_sum)
        return log_likelihood, alpha

    def _beta_recursion(self, trellis):
        T = self.num_frames
        beta = np.zeros(trellis.shape);
        beta[:, -1] = np.ones(self.num_states)
        for t in range(T - 1)[::-1]:
            beta[:, t] = np.dot(self.A, (trellis[:, t + 1] * beta[:, t + 1]))
            beta[:, t] /= np.sum(beta[:, t])
        return beta

    #todo: check on this
    def _gamma_calc(self, alpha, beta):
        gamma = np.zeros((self.num_states, self.num_frames))
        for t in range(self.num_frames - 1):
            partial_g = alpha[:, t] * beta[:, t]
            gamma[:, t] = normalize(partial_g)
        partial_g = alpha[:, -1] * beta[:, -1]
        gamma[:, -1] = normalize(partial_g)
        return gamma

    def _expectation_maximization(self, mfcc):
        trellis = self._trellis_init(mfcc)
        T = self.num_frames

        log_likelihood, alpha = self._alpha_recursion(mfcc,trellis)
        beta = self._beta_recursion(trellis)
        gamma = self._gamma_calc(alpha, beta)

        xi_sum = np.zeros((self.num_states, self.num_states))
        new_A = np.zeros((self.num_states, self.num_states))
        for t in range(T - 1):
            partial_sum = self.A * np.dot(alpha[:, t], (beta[:, t] * trellis[:, t + 1]).T)
            xi_sum += normalize(partial_sum)

        gamma_state_sum = np.sum(gamma, axis=1)

        for i in range(self.num_states):
            new_A[:,i] = xi_sum[i] / gamma_state_sum[i]

        expected_prior = gamma[:, 0]
        expected_A = new_A

        expected_mu = np.zeros((self.n_dims, self.num_states))
        expected_covs = np.zeros((self.n_dims, self.n_dims, self.num_states))

        #Set zeros to 1 before dividing
        gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)

        for s in range(self.num_states):
            gamma_obs = mfcc * gamma[s, :]
            expected_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]
            numerator_sum = np.zeros((self.n_dims,self.n_dims))
            for t in range(self.num_frames):
                numerator_sum += np.diag((mfcc[:,t] - expected_mu[:,s]) * (mfcc[:,t] - expected_mu[:,s].T)) * gamma[s,t]
            expected_covs[:,:,s] = numerator_sum / gamma_state_sum[s]

        self.prior = expected_prior
        self.mu = expected_mu
        self.covs = expected_covs
        self.A = expected_A
        print (log_likelihood)
        return (expected_prior, expected_A, expected_mu, expected_covs)

    def train(self, num_training_samples):
        self.L = num_training_samples
        prior_sum = None
        A_sum = None
        mu_sum = None
        covs_sum = None
        for l in range(self.L):
            #load sample
            fp = input("Enter wave filepath or filename")
            FS, signal = scipy.io.wavfile.read(fp)

            #silence processing
            sig = find_speech(signal,FS)
            #get mfccs
            mfcc = ret_mfcc(sig,FS)

            if l == 0:
                prior_sum, A_sum, mu_sum, covs_sum = self._expectation_maximization(mfcc)
            else:
                ret = _expectation_maximization(mfcc)
                prior_sum += ret[0]
                A_sum += ret[1]
                mu_sum += ret[2]
                covs_sim += ret[3]

        self.prior = prior_sum / self.L
        self.A = A_sum / self.L
        self.mu = mu_sum / self.L
        self.covs = covs_sum / self.L
        np.save(self.name + '_inital_state', self.prior)
        np.save(self.name + '_A', self.A)
        np.save(self.name + '_mu', self.prior)
        np.save(self.name + '_covs', self.prior)

    def load_lambda(self):
        try:
            self.prior = load(self.name + '_inital_state.npy')
            self.A = load(self.name + '_A.npy')
            self.mu = load(self.name + '_mu.npy')
            self.covs = load(self.name + '_covs.npy')
        except:
            print ("Loading failed.")


if __name__ == "__main__":
    #TEST
    # rstate = np.random.RandomState(0)
    # t1 = np.ones((4, 40)) + .001 * rstate.rand(4, 40)
    # t1 /= t1.sum(axis=0)
    m1 = HMM('test', 2)
    m1.train(1)
    m1.train(1)
    # trellis = m1._trellis_init(t1)
    # ll,alpha =  m1._alpha_recursion(t1, trellis)
    # beta =  m1.__beta_recursion(trellis)
    # gamma = m1._gamma_calc(alpha,beta)
    #m1 = HMM()
