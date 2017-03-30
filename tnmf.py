# -*- coding: utf-8 -*-
"""
tnmf.py
~~~~~~~
.. topic:: Contents

    The tnmf module is used to perform task-driven nonnegative
    matrix factorisation (TNMF).
    It includes the SupervisedDL class, fit and score methods

    Created on Wed Jun 29 16:37:28 2016

    @authors: bisot, serizel
    
    Copyright 2016-2017 Victor Bisot, Romain Serizel
    This software is distributed under the terms of the GNU Public License
    version 3 (http://www.gnu.org/licenses/gpl.txt)

    .. [#] V.Bisot, R. Serizel, S. Essid, and G. Richard.
        "Feature Learning with Matrix Factorization Applied to
        Acoustic Scene Classification".
        Accepted for publication in *IEEE Transactions on Audio,
        Speech and Language Processing*, 2017
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
import copy


class SupervisedDL(object):

    """ Supervised DL class

    Task-driven Dictionary Learning with modified algorithm

    Parameters
    ----------
    data : array, shape (n_samples, n_features)
        Training data matrix
        Needs to be provided if initialization is done in the model

    n_components : int
        Size of the dictionary

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

    batch_size : int, optional
        Size of the batch (1 for stochastic gradient)

    verbose : int
        Set verbose to any positive number for verbosity.

    Attributes
    ----------
    clf : object
        Classifier

    D : array
        Dictionnary

    """
    def __init__(self, data=np.asarray([[0, 0]]), n_components=64,
                 n_labels=2, pos=True, n_iter=1,
                 lambda1=0, lambda2=0, rho=0.001, verbose=0, mu=1, agreg=1,
                 init='random', n_iter_init=10, batch_size=6250,
                 max_iter_init=10, max_iter_inloop=1):

        self.data = data
        self.data_shape = data.shape
        self.n_components = n_components
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
        self.clf = LogisticRegression(
            C=self.mu, multi_class='multinomial',
            solver='lbfgs', max_iter=self.max_iter_init, warm_start=True)

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
                print("""Initializing dictionnary
                      with sklearn DictionnaryLearning""")
            u_dl = decomposition.DictionaryLearning(
                n_components=self.n_components,
                alpha=0, max_iter=n_iter_init)
            u_dl.fit(self.data)
            self.D = preprocessing.normalize(u_dl.components_.T, axis=0)

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
            tic = time.time()
            # Average projections if necessary
            alpha_mean = self.mean_frames(
                alpha_mat.toarray().T, agreg=self.agreg)
            alpha_mean = preprocessing.scale(alpha_mean, with_mean=False)
            # Update classifier
            self.clf.fit(alpha_mean, y_train)
            self.w = self.clf.coef_
            self.b = self.clf.intercept_

            if i == 0:
                # Classifier is initialized on first iteration
                # For further iterations, the LR class is only updated on
                # 1 iteration with warm restart
                self.clf.max_iter = self.max_iter_inloop

            # Print current performance #
            if self.verbose > 0:  # Print the scores
                print("Iteration number %i \n" % i)
                if X_test.any():
                    a, f1, = self.scores(X=X_test, y=y_test)
                    print("""Classification scores on test set:
                          a=%0.3f   f1=%0.3f""" % (a, f1))
                a, f1, = self.scores(X=X_train, y=y_train)
                print("""Classification scores on train set:
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
                alpha_mat = spams.lasso(np.asfortranarray(x_mat),
                                        D=np.asfortranarray(self.D),
                                        lambda1=self.lambda1,
                                        lambda2=self.lambda2, mode=2,
                                        pos=self.pos)
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
                            self.w[k, :][:, np.newaxis], tmp[:, np.newaxis].T))
                    d_alpha = (
                        num_alpha.T / denom[:, np.newaxis]) - self.w[y, :]

                    # Update D
                    self.update_D(
                        x_mat=x_mat, y=y, denom=denom,
                        num_alpha=num_alpha, d_alpha=d_alpha,
                        alpha_mat=alpha_mat, rho_t=rho_t)
        # Print final performance #

        if self.verbose > 0:  # Print the scores
            print("Final model")
            if X_test.any():
                a, f1 = self.scores(X=X_test, y=y_test)
                print("""Classification scores on test set:
                      a=%0.3f   f1=%0.3f""" % (a, f1))
            a, f1 = self.scores(X=X_train, y=y_train)
            print("""Classification scores on train set:
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

        alpha_mean = self.mean_frames(alpha_mat.toarray().T, agreg=self.agreg)
        alpha_mean = preprocessing.scale(alpha_mean, with_mean=False)
        y_pred = self.clf.predict(alpha_mean)
        a = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        return a, f1

    def update_D(self, x_mat=0, y=0, alpha_mat=0, denom=0, num_alpha=0,
                 d_alpha=0, rho_t=0.001):
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



        """
        non_zero = alpha_mat.nonzero()
        beta = np.zeros(num_alpha.shape)
        for i in range(num_alpha.shape[1]):
            ind = non_zero[0][non_zero[1] == i]
            if ind.shape[0] > 1:
                beta[ind, i] = np.dot(
                    spams.invSym(
                        np.asfortranarray(
                            np.dot(
                                np.transpose(self.D[:, ind]),
                                self.D[:, ind]) +
                            self.lambda2)),
                    d_alpha[i, ind])
            elif ind.shape[0] == 1:
                beta[ind, i] = np.dot(
                    1./(np.dot(np.transpose(self.D[:, ind]),
                               self.D[:, ind]) + self.lambda2),
                    d_alpha[i, ind])

        alpha_mat = alpha_mat.toarray()

        d_D = np.dot((x_mat-np.dot(self.D, alpha_mat)), np.transpose(beta))
        d_D -= np.dot(np.dot(self.D, beta), np.transpose(alpha_mat))
        self.D = self.D - rho_t * d_D

        if self.pos is True:
            self.D[np.where(self.D < 0)] = 0.0000001
        self.D = preprocessing.normalize(self.D, axis=0)
