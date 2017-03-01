# -*- coding: utf-8 -*-
"""
tgnmf.py
~~~~~~~~
.. topic:: Contents

    The tgnmf module is used to perform task-driven group nonnegative
    matrix factorisation (TGNMF).
    It includes the SupervisedDL class, fit and score methods

    Created on Wed Jun 29 16:37:28 2016

    @authors: bisot, serizel

    .. [#] V.Bisot, R. Serizel, S. Essid, and G. Richard.
        "Feature Learning with Matrix Factorization Applied to
        Acoustic Scene Classification".
        Accepted for publication in *IEEE Transactions on Audio,
        Speech and Language Processing*, 2017

    .. [#] R. Serizel, V.Bisot, S. Essid, and G. Richard.
        “Supervised group nonnegative matrix factorisation with similarity
        constraints and applications to speaker identification”.
        In Proc. of *2017 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP)*, 2017.
"""

import numpy as np
import spams
from sklearn import decomposition
import beta_ntf
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,  f1_score
from sklearn.linear_model import LogisticRegression
import random
import time

from sklearn.utils.extmath import (
    logsumexp, log_logistic, safe_sparse_dot,
    softmax, squared_norm)
from sklearn.utils.validation import (
    DataConversionWarning,
    check_X_y, NotFittedError)
from sklearn.preprocessing import LabelEncoder, LabelBinarizer


class SupervisedDL(object):
    """ Supervised DL class

    Task-driven Dictionary Learning with modified algorithm

    Parameters
    ----------
    data : array, shape (n_samples, n_features)
        Training data matrix
        Needs to be provided if initialization is done in the model

    n_labels : int
        number of classes

    pos : bool, default: True
        When set to True, the model is fit in its nonnegative formulation

    n_iter : int
        Number of epochs on which to run the algorithm

    lambda1 : float, default: 0.1
        Paramter controlling the l1 norm penality for the projection step

    lambda2 : float, default: 0
        Paramter controlling the l2 norm penality for the projection step

    rho : float, default: 0.001
        Initial gradient step parameter

    mu : float, default: 1
        Regularization strenght of the classifier; must be positive.
        Smaller values specify stronger regularization.

    agreg :  int, optional (default: 1)
        In the case classification is done on bags of agreg successive
        projections. Every successive group of agreg  projections
        (without overlapp) will be averaged before classification

    init :  str, {'random', 'nmf', 'dic-learning'}
        Controls the nature of the dictionary initialization.

        * Use 'random' for a random initalization of
        * Use 'nmf' for intializaing with nonnegative matrix factorization
        * Use 'dic-learning' to intilaize with the scikit-learn DictionaryLearning class

    n_iter_init : int, optional
        Number of iterations for the dictionary initialization

    max_iter_init : int, optional
        Maximum number of iterations for the classifier initialization

    max_iter_inloop : int, optional
        Maximum number of iterations at each epoch for the classifier update

    max_iter_fin : int, optional
        Maximum number of iterations once the dictionnary is learnt

    batch_size : int, optional
        Size of the batch (1 for stochastic gradient)

    lbl : array (n_couples, 2)
        Unique (class, session) couples (See also [3]_)

        * 'lbl[:, 0]': class labels
        * 'lbl[:, 1]': session labels

        This is equivalent to the attribute 'self.iters['cls']' in GNMF_

    ses_train : array (n_samples, 1)
        Session labels for the data

    sub_dict_size : int
        Size of the sub-dictionnaries related to a unique couple (cls, ses)
        See also [3]_

    k_cls : int
        Number of components that are affected to cls related bases.
        See also [3]_

    k_ses : int
        Number of components that are affected to ses related bases.
        See also [3]_

    nu1 : float
        Class similarity constraint
        See also [3]_

    nu2 : float
        Session similarity constraint
        See also [3]_

    verbose : int
        Set verbose to any positive number for verbosity.

    Attributes
    ----------
    clf : object
        Classifier

    D : array
        Dictionnary

    n_components : int
        Size of the dictionary

    dist_ses : array
        Distance between bases related to the same session

    dist_cls :
        Distance between bases related to the same class

    cst : array
        Update constraint computed from dist_ses and dist_cls


    References
    ----------

    .. [#] R. Serizel, S. Essid, and G. Richard.
        “Group nonnegative matrix factorisation with speaker and session
        variability compensation for speaker identification”.
        In Proc. of *2016 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP)*, pp. 5470-5474, 2016.

    .. _GNMF: http://rserizel.github.io/groupNMF/_modules/beta_nmf_class.html#ClassBetaNMF
     """

    def __init__(self, data=np.asarray([[0, 0]]),
                 n_labels=2, pos=True, n_iter=1,
                 lambda1=0, lambda2=0, rho=0.001, verbose=0, mu=1, agreg=1,
                 init='random', n_iter_init=10, batch_size=6250,
                 max_iter_init=10, max_iter_inloop=1, max_iter_fin=0,
                 lbl=np.asarray([[0, 0]]), ses_train=0,
                 sub_dict_size=1, k_cls=0, k_ses=0, nu1=0, nu2=0):

        self.data = data
        self.data_shape = data.shape
        self.verbose = verbose
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.n_iter = n_iter
        self.n_labels = n_labels
        self.init = init
        self.rho = rho
        self.mu = mu
        self.agreg = agreg
        self.pos = pos
        self.analysis = np.zeros(shape=(n_iter, 5))
        self.batch_size = batch_size
        self.max_iter_init = max_iter_init
        self.max_iter_inloop = max_iter_inloop
        self.max_iter_fin = max_iter_fin
        self.clf = LogisticRegression(
            C=self.mu, multi_class='multinomial',
            solver='lbfgs', max_iter=self.max_iter_init, warm_start=True)
        self.lbl = lbl
        self.sub_dict_size = sub_dict_size
        self.k_cls = k_cls
        self.k_ses = k_ses
        self.n_components = lbl.shape[0]*(k_cls + k_ses)
        self.nu1 = nu1
        self.nu2 = nu2
        self.ses_train = ses_train

        if self.init == 'random':
            if self.pos is False:
                self.D = np.random.randn(self.data_shape[1],
                                         self.n_components)
            if self.pos is True:
                self.D = np.abs(np.random.randn(self.data_shape[1],
                                                self.n_components))
            self.D = preprocessing.normalize(self.D, axis=0)

        if self.init == 'NMF':
            if self.verbose > 0:
                print("Initializing dictionnary with beta_ntf")
            ntf = beta_ntf.BetaNTF(data_shape=self.data_shape,
                                   n_components=self.n_components,
                                   n_iter=n_iter_init, verbose=False, beta=2)
            ntf.fit(self.data)
            self.D = ntf.factors_[1]
            self.D = preprocessing.normalize(self.D, axis=0)

        if self.init == 'DictionnaryLearning':
            if self.verbose > 0:
                print(""" Initializing dictionnary
                          with sklearn DictionnaryLearning""")
            u_dl = decomposition.DictionaryLearning(
                n_components=self.n_components,
                alpha=0, max_iter=n_iter_init)
            u_dl.fit(self.data)
            self.D = preprocessing.normalize(
                u_dl.components_.T, axis=0)
        self.compute_cst()

    def compute_cst(self):
        """ Compute the update constraints based on the distance between
        session related bases and the distance between class related bases"""

        lbl = self.lbl
        cls_card = np.zeros((int(np.amax(lbl[:, 0])), ))
        cls_sum = np.zeros((int(np.amax(lbl[:, 0])),
                            self.D.shape[0], self.k_cls))
        self.dist_cls = 0
        for i in range(len(cls_card)):
            cls_card[i] = max(lbl[lbl[:, 0] == i].shape[0] - 1, 0)
            ref_sub_ind = np.arange(i*self.sub_dict_size,
                                    i*self.sub_dict_size + self.k_cls)
            for j in np.where(lbl[lbl[:, 0] == i])[0]:
                sub_ind = np.arange(j*self.sub_dict_size,
                                    j*self.sub_dict_size + self.k_cls)
                cls_sum[i, ] += self.D[:, sub_ind]
                self.dist_cls += np.linalg.norm(
                    self.D[:, sub_ind] - self.D[:, ref_sub_ind])

        ses_card = np.zeros((int(np.amax(lbl[:, 1])), ))
        ses_sum = np.zeros((int(np.amax(lbl[:, 1])),
                            self.D.shape[0], self.k_ses))
        self.dist_ses = 0
        for i in range(len(ses_card)):
            ses_card[i] = max(lbl[lbl[:, 1] == i].shape[0] - 1, 0)
            ref_sub_ind = np.arange(
                i*self.sub_dict_size + self.k_cls,
                i*self.sub_dict_size + self.k_cls + self.k_ses)
            for j in np.where(lbl[lbl[:, 1] == i])[0]:
                sub_ind = np.arange(
                    j*self.sub_dict_size + self.k_cls,
                    j*self.sub_dict_size + self.k_cls + self.k_ses)
                ses_sum[i, ] += self.D[:, sub_ind]
                self.dist_ses += np.linalg.norm(
                    self.D[:, sub_ind] - self.D[:, ref_sub_ind])

        self.cst = np.zeros((self.D.shape))
        for i in range(lbl.shape[0]):
            cls = int(lbl[i, 0]-1)
            ses = int(lbl[i, 1]-1)
            sub_ind = np.arange(i*self.sub_dict_size,
                                (i+1)*self.sub_dict_size)
            D_sub = self.D[:, sub_ind]
            sub_cst = np.zeros((self.D.shape[0], self.sub_dict_size))

            sub_cst[:, :self.k_cls] = self.nu1 * (
                cls_card[cls] * D_sub[:, :self.k_cls] -
                (cls_sum[cls, ] - D_sub[:, :self.k_cls]))
            sub_cst[:, self.k_cls: self.k_cls+self.k_ses] = self.nu2 * (
                ses_card[ses] *
                D_sub[:, :self.k_cls:self.k_cls+self.k_ses] -
                (ses_sum[ses] - D_sub[:, :self.k_cls:self.k_cls+self.k_ses]))
            self.cst[:, sub_ind] = sub_cst

    def mean_frames(self, X=0, agreg=15):
        """
        Averages every successive group of agreg rows in the a matrix

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Matrix to reduce by averaging

        agreg : int
            Specifies size of the groups to average

        Returns
        -------
        X_mean: array, shape(n_samples/agreg, n_features)
            Averaged matrix
        """
        return np.mean(np.reshape(X, (X.shape[0]/agreg, agreg, -1)), axis=1)

    def project_data(self, X, agreg=1):
        """
        Projects data on the model dictionary

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Matrix to project on dictoonary

        agreg : int
            Specifies size of the groups to average after projection

        Returns
        -------
        projections: array, shape(n_samples/agreg, n_components)
            Projection matrix
        """
        alpha_mat = spams.lasso(np.asfortranarray(np.transpose(X)),
                                D=np.asfortranarray(self.D),
                                lambda1=self.lambda1,
                                lambda2=self.lambda2, mode=2,
                                pos=self.pos)
        alpha_mat = alpha_mat.toarray()
        if agreg > 1:
            return np.mean(
                np.reshape(
                    alpha_mat,
                    (alpha_mat.shape[0]/agreg, agreg, -1)),
                axis=1)
        else:
            return alpha_mat

    def predict(self, X):
        """
        Predicts labels from a given matrix using the model classifier

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Matrix to project on dictoonary

        Returns
        -------
        y_pred: array, shape(n_samples/agreg,)
            Predicted labels
        """
        alpha_mat = np.transpose(self.project_data(X=X, agreg=self.agreg))
        y_pred = self.clf.predict(alpha_mat)

        return y_pred

    def fit(self, X_train, y_train, X_test=np.array([]), y_test=np.array([])):
        """
        Fits the model to input training data

        Parameters
        ----------
        X_train : array, shape (n_samples, n_features)
            Training data matrix

        X_test : array, shape (n_samples_test, n_features), optional
            Test data matrix
            Usefull only to check performance during development

        y_train : array-like, shape (n_samples/self.agreg,)
            Target vector relative to X_train.

        y_train : array-like, shape (n_samples_test/self.agreg,)
            Target vector relative to X_test.

        Returns
        -------

        """
        for i in range(self.n_iter):

            # Classifier update step #

            tic = time.time()
            # Project full data on D
            alpha_mat = spams.lasso(
                np.asfortranarray(X_train.T),
                D=np.asfortranarray(self.D),
                lambda1=self.lambda1, lambda2=self.lambda2,
                mode=2, pos=self.pos)
            # Average projections if necessary
            alpha_mean = self.mean_frames(
                alpha_mat.toarray().T, agreg=self.agreg)
            alpha_mean = preprocessing.scale(alpha_mean, with_mean=False)

            # Update classifier
            self.clf.fit(alpha_mean, y_train)
            self.w = self.clf.coef_
            self.b = self.clf.intercept_
            self. compute_cst()
            if i == 0:
                # Classifier is initialized on first iteration
                # For further iterations, the LR class is only updated on
                # 1 iteration with warm restart
                self.clf.max_iter = self.max_iter_inloop

            # Print current performance #

            if self.verbose > 0:  # Print the scores
                print("Iteration number %i \n" % i)
                X2, y = check_X_y(
                    alpha_mean, y_train, accept_sparse='csr',
                    dtype=np.float64, order="C")
                lbin = LabelBinarizer()
                Y_binarized = lbin.fit_transform(y_train)
                yo = np.zeros(shape=(self.n_labels, 1))
                yo[:, 0] = self.clf.intercept_
                yo = np.concatenate((self.clf.coef_, yo), axis=1)
                if Y_binarized.shape[1] == 1:
                    Y_binarized = np.hstack([1 - Y_binarized, Y_binarized])
                ut = _multinomial_loss(
                    yo, X2,  Y_binarized, 1/self.mu,
                    sample_weight=np.ones(y.shape[0]))
                yo = np.zeros(shape=(self.n_labels, 1))
                print "Classification costs: ", ut[0]
                print(
                    "Class distance: ", self.dist_cls,
                    "Session distance:", self.dist_ses)
                if i == 0:
                    self.nu1 *= ut[0]/self.dist_cls
                    self.nu2 *= ut[0]/self.dist_ses
                    print "Weights nu1 and nu2", self.nu1, self.nu2
                    self.compute_cst()
                if X_test.any():
                    a, f1 = self.scores(X=X_test, y=y_test)
                    print(""" Classification scores on test set:
                              a=%0.3f   f1=%0.3f""" % (a, f1))
                a, f1 = self.scores(X=X_train, y=y_train)
                print(""" Classification scores on train set:
                          a=%0.3f   f1=%0.3f""" % (a, f1))

            # Dictionary update step #

            draw = range(self.agreg*len(y_train))
            random.shuffle(draw)
            nb_batch = len(draw)/int(self.batch_size) + 1
            for t in range(nb_batch):
                # Select and project a data point

                ind = draw[
                    t*self.batch_size: min((t+1)*self.batch_size, len(draw))]
                x_mat = np.transpose(X_train[ind, ])
                ind = [x/self.agreg for x in ind]
                y = y_train[ind]
                ses = self.ses_train[ind]
                alpha_mat = spams.lasso(np.asfortranarray(x_mat),
                                        D=np.asfortranarray(self.D),
                                        lambda1=self.lambda1,
                                        lambda2=self.lambda2,
                                        mode=2, pos=self.pos)
                alpha_mean = alpha_mat.toarray()

                if alpha_mean.nonzero()[0].any():
                    # Step decaying heuristic
                    rho_t = np.min(
                        [
                            self.rho,
                            self.rho * (
                                nb_batch*self.n_iter*self.batch_size
                                ) / (
                                10*(
                                    (i*nb_batch*self.batch_size) +
                                    t*self.batch_size + 1))])

                    # Gradient of loss with respect to projections
                    denom = np.zeros((alpha_mat.shape[1], ))
                    num_alpha = np.zeros(alpha_mat.shape)
                    for k in range(self.n_labels):
                        tmp = np.exp(
                            np.dot(self.w[k, :], alpha_mean) + self.b[k])
                        denom += tmp
                        num_alpha += (np.dot(
                            self.w[k, :][:, np.newaxis],
                            tmp[:, np.newaxis].T))
                    d_alpha = (
                        num_alpha.T / denom[:, np.newaxis]) - self.w[y, :]

                    # Update D
                    self.update_D(
                        x_mat=x_mat, y=y, denom=denom,
                        num_alpha=num_alpha, d_alpha=d_alpha,
                        alpha_mat=alpha_mat, rho_t=rho_t, ses=ses)

            if self.verbose > 0:
                print "Iteration time", time.time() - tic
        if self.max_iter_fin > 0:
            alpha_mat = spams.lasso(
                np.asfortranarray(X_train.T),
                D=np.asfortranarray(self.D),
                lambda1=self.lambda1, lambda2=self.lambda2,
                mode=2, pos=self.pos)
            # Average projections if necessary
            alpha_mean = self.mean_frames(
                alpha_mat.toarray().T, agreg=self.agreg)
            alpha_mean = preprocessing.scale(alpha_mean, with_mean=False)

            # Update classifier
            self.clf.max_iter = self.max_iter_fin
            self.clf.fit(alpha_mean, y_train)
        # Print final performance #

        if self.verbose > 0:  # Print the scores
            print("Final model")
            if X_test.any():
                a, f1 = self.scores(X=X_test, y=y_test)
                print(""" Classification scores on test set:
                          a=%0.3f   f1=%0.3f""" % (a, f1))
            a, f1, = self.scores(X=X_train, y=y_train)
            print(""" Classification scores on train set:
                      a=%0.3f   f1=%0.3f""" % (a, f1))

    def scores(self, X, y):
        """
        Compute classification scores (accurracy and F1-score).
        on a given dataset.

        Parameters
        ----------
        X_train : array, shape (n_samples, n_features)
            Training data matrix

        X_test : array, shape (n_samples_test, n_features), optional
            Test data matrix
            Usefull only to check performance during development

        y_train : array-like, shape (n_samples/self.agreg,)
            Target vector relative to X_train.

        y_train : array-like, shape (n_samples_test/self.agreg,)
            Target vector relative to X_test.

        Returns
        -------
        y_pred: array, shape(n_samples/agreg,)
        """
        tic = time.time()
        alpha_mat = spams.lasso(
            np.asfortranarray(X.T),
            D=np.asfortranarray(self.D),
            lambda1=self.lambda1,
            lambda2=self.lambda2, mode=2,
            pos=self.pos)

        alpha_mean = self.mean_frames(
            alpha_mat.toarray().T, agreg=self.agreg)
        alpha_mean = preprocessing.scale(
            alpha_mean, with_mean=False)
        y_pred = self.clf.predict(alpha_mean)
        a = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')

        return a, f1

    def update_D(self, x_mat=0, y=0, alpha_mat=0, denom=0, num_alpha=0,
                 d_alpha=0, rho_t=0.001, ses=0):
        """
        Updates dictionary

        Parameters
        ----------
        x_mat : array shape (n_features, batch_size)
            Input data (batch_size = 1 in the stochastic gradient case)

        y : array shape (batch_size, )
            Labels corresponding to the input data

        alpha_mat : array shape (batch_size, n_components)
            Projections

        d_alpha : array shape (batch_size, n_components)
            Gradient of loss with respect to projections

        denom : array shape (n_components, )
            Gradient denominator

        num_alpha : array (batch_size, n_components)
            Gradient numerator

        rho_t : float
            Learning rate

        ses : array (batch_size, )
            Session labels for the data

        """
        non_zero = alpha_mat.nonzero()
        beta = np.zeros(num_alpha.shape)
        beta_sub = np.zeros(num_alpha.shape)
        n_nan = 0
        tot_loop = 0
        lr_weight = np.zeros((beta.shape[0], ))
        alpha_mat = alpha_mat.toarray()
        alpha_mat_sub = np.zeros(alpha_mat.shape)
        for i in range(num_alpha.shape[1]):
            cls_ind = np.where(
                ((self.lbl[:, 0] == y[i]) & (self.lbl[:, 1] == ses[i])))[0][0]
            sub_ind = np.arange(cls_ind*self.sub_dict_size,
                                (cls_ind+1)*self.sub_dict_size)
            ind = non_zero[0][non_zero[1] == i]
            lr_weight[sub_ind] += 1
            alpha_mat_sub[sub_ind, i] = alpha_mat[sub_ind, i]
            if sub_ind.shape[0] > 1:
                tot_loop += 1
                beta[ind, i] = np.dot(
                    spams.invSym(
                        np.asfortranarray(
                            np.dot(
                                np.transpose(self.D[:, ind]),
                                self.D[:, ind]) +
                            self.lambda2)),
                    d_alpha[i, ind])
                if np.isnan(np.amax(np.abs(beta[sub_ind, i]))):
                    n_nan += 1
                    beta[ind, i] = np.zeros(beta[sub_ind, i].shape)
                beta_sub[sub_ind, i] = beta[sub_ind, i]
            elif sub_ind.shape[0] == 1:
                tot_loop += 1
                beta[sub_ind, i] = np.dot(
                    1./(np.dot(np.transpose(self.D[:, sub_ind]),
                               self.D[:, sub_ind]) + self.lambda2),
                    d_alpha[i, ind])
                alpha_mat_sub[sub_ind, i] = alpha_mat[sub_ind, i]
        if n_nan > 0:
            if tot_loop/(n_nan+1) < 2:
                print "Warning nan occurence", n_nan, "total loops", tot_loop
        d_D = np.dot(
            (x_mat-np.dot(self.D, alpha_mat_sub)), np.transpose(beta_sub))
        d_D -= np.dot(np.dot(self.D, beta_sub), np.transpose(alpha_mat_sub))

        sub_ind = np.arange(cls_ind*self.sub_dict_size,
                            (cls_ind+1)*self.sub_dict_size)
        rho_t *= lr_weight/self.batch_size
        self.D = self.D - rho_t * d_D - self.cst

        if self.pos is True:
            self.D[np.where(self.D < 0)] = 0.0000001
        self.D = preprocessing.normalize(self.D, axis=0)


def _multinomial_loss(w, X, Y, alpha, sample_weight):
    """Computes multinomial loss and class probabilities.
    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    Returns
    -------
    loss : float
        Multinomial loss.
    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities.
    w : ndarray, shape (n_classes, n_features)
        Reshaped param vector excluding intercept terms.
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))
    w = w.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * p).sum()
    loss += 0.5 * alpha * squared_norm(w)
    p = np.exp(p, p)
    return loss, p, w


def _intercept_dot(w, X, y):
    """Computes y * np.dot(X, w).
    It takes into consideration if the intercept should be fit or not.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    """
    c = 0.
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]

    z = safe_sparse_dot(X, w) + c
    return w, c, y * z
