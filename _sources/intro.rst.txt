Supervised group NMF
--------------------

In [1]_, we adapted a supervised matrix factorization model known as *Task-driven Dictionary Learning* (TDL) [3]_} to suit the ASC task. We introduced a variant of the TDL model in its nonnegative formulation [4]_, including a modification of the original algorithm, where a nonnegative dictionary is jointly learned with a multi-class classifier. In [2]_ we proposed a new formulation of the Group nonnegative matrix factorisation (GNMF_) method. Using the Euclidean distance as the divergence for the GNMF_ problem, the dictionary learning based on GNMF_ is integrated in a supervised framework inspired by TDL [3]_.

GNMF with speaker and session similarity
++++++++++++++++++++++++++++++++++++++++

Details_ and `source code`_ for the GNMF [3]_ .

.. _Details: http://rserizel.github.io/groupNMF/intro.html
.. _source code: https://github.com/rserizel/groupNMF

Task-driven NMF based dictionary learning
+++++++++++++++++++++++++++++++++++++++++
TDL [4]_ has recently been applied with nonnegativity constraints to perform speech enhancement[5]_ or to acoustic scene classification, where temporally integrated projections are classified with multinomial logistic regression [1]_. In [2]_ we extended the latter approach to the GNMF_ case.

Task-driven NMF
~~~~~~~~~~~~~~~

The general idea of nonnegative TDL or task-driven NMF (TNMF) is to unite the dictionary learning with NMF and the training of the classifier in a joint optimization problem [1]_, [5]_. Influenced by the classifier, the basis vectors are encouraged to explain the discriminative information in the data while keeping a low reconstruction cost. The TNMF model first considers the optimal projections :math:`\textbf{h}^{\star}(\textbf{v},\textbf{W})` of the data points :math:`\textbf{v}` on the dictionary \textbf{W}, which are defined as solutions of the nonnegative elastic-net problem [6]_, expressed as:

.. math::
	\textbf{h}^{\star}(\textbf{v},\textbf{W}) = \min_{\textbf{h} \in \mathbb{R}_+^{K}}\ \frac{1}{2}\|\textbf{v}-\textbf{W}\textbf{h}\|_{2}^{2}+\lambda_{1}\|\textbf{h}\|_{1}+\frac{\lambda_{2}}{2}\|\textbf{h}\|_{2}^{2};
	:label: elastic

where :math:`\lambda_{1}` and :math:`\lambda_{2}` are nonnegative regularization parameters. Given each data segment :math:`\textbf{V}^{(l)}` of length :math:`M` frames, associated with a label :math:`y` in a fixed set of labels :math:`\mathcal{Y}`, we want to classify the mean of the projections of the data points :math:`\textbf{v}^{(l)}` belonging to the segment :math:`l`, such that :math:`\textbf{V}^{(l)}=[\textbf{v}_{0}^{(l)},...,\textbf{v}_{M-1}^{(l)}]`. We define :math:`\hat{\textbf{h}}^{(l)}` as the averaged projection of :math:`\textbf{V}^{(l)}` on the dictionary, where :math:`\hat{\textbf{h}}^{(l)}=\frac{1}{M}\sum_{m=0}^{M-1}\textbf{h}^{\star}(\textbf{v}_{m}^{(l)},\textbf{W})`. The corresponding classification loss (here using multinomial logistic regression) is defined as :math:`l_{s}(y,\textbf{A},\hat{\textbf{h}}^{(l)})`, where :math:`\textbf{A}\in \mathcal{A}` are the parameters of the classifier. The TNMF problem is then expressed as a joint minimization of the expected classification loss over :math:`\textbf{W}` and :math:`\textbf{A}`:

.. math::
	\min_{\textbf{W} \in \mathcal{W}, \textbf{A} \in \mathcal{A}} f(\textbf{W},\textbf{A})+\frac{\nu}{2}\|\textbf{A}\|_{2}^{2},

with

.. math::
	f(\textbf{W},\textbf{A}) = \mathbb{E}_{y,\textbf{V}^{(l)}}\ [l_{s}(y,\textbf{A},\hat{\textbf{h}}^{(l)}(\textbf{v}^{(l)},\textbf{W}))]. 
	:label: TDL

Here, :math:`\mathcal{W}` is defined as the set of nonnegative dictionaries containing unit :math:`l_{2}`-norm basis vectors and :math:`\nu` is a regularization parameter on the classifier parameters, meant to prevent over-fitting. The problem in equation :eq:`TDL` is optimized with mini-batch stochastic gradient descent as described in the paper of Bisot *et al.* [1]_.
  
Task-driven GNMF
~~~~~~~~~~~~~~~~
In task-driven GNMF_ (TGNMF) we propose to perform jointly the dictionary learning based on GNMF_ [2]_ and the training of a multinomial logistic regression. The dictionary :math:`\textbf{W}` is then the concatenation of all the sub-dictionaries :math:`\textbf{W}^{(cs)}` and the optimal projections :math:`\textbf{h}^{\star}(\textbf{v},\textbf{W})` are the solutions of :eq:`elastic`.

Including the `similarity constraints`_ , the TGNMF is thus expressed as the minimization of the following problem:

.. math::
	\min_{\textbf{W}\in \mathcal{W}, \textbf{A} \in \mathcal{A}} f(\textbf{W},\textbf{A})+\frac{\mu}{2}\|\textbf{A}\|_{2}^{2} + \nu_1 J_{\mathrm{SPK}} + \nu_2 J_{\mathrm{SES}},

with :math:`f(\textbf{W},\textbf{A})` as defined above. The problem is again optimized with mini-batch stochastic gradient descent. However, as opposed to the previous algorithm, for each data point :math:`\textbf{v}` belonging to a particular :math:`\textbf{V}^{(cs)}`, only the corresponding sub-dictionaries (:math:`\textbf{W}^{(cs)}`) are updated, whereas the other dictionaries are left unchanged in order to match the GNMF_ adaptation scheme [2]_.

Download
++++++++

Source code available at https://github.com/rserizel/TGNMF

Getting Started
+++++++++++++++
A short example is available as at https://github.com/rserizel/TGNMF/blob/master/TGNMF_howto.ipynb

Citation
++++++++

If you are using this source code please consider citing one of the following papers: 


.. topic:: References

	.. [#] V. Bisot, R. Serizel, S. Essid, and G. Richard. "Feature Learning with Matrix Factorization Applied to Acoustic Scene Classification". Accepted for publication in *IEEE Transactions on Audio, Speech and Language Processing*, 2017
	.. [#] R. Serizel, V.Bisot, S. Essid, and G. Richard. "Supervised group nonnegative matrix factorisation with similarity constraints and applications to speaker identification". In Proc. of *2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2017.

	::

		@article{bisot2017TASLP,
	  	title={Feature Learning with Matrix Factorization Applied to Acoustic Scene Classification},
	  	author={Serizel, Romain and Bisot, Victor and Essid, Slim and Richard, Ga{\"e}l},
	  	journal={IEEE Transactions on Audio, Speech and Language Processing},
	  	pages={14},
	  	year={2017},
	  	organization={IEEE}
		}

	
	::

		@inproceedings{serizel2017ICASSP,
	  	title={Supervised group nonnegative matrix factorisation with similarity constraints and applications to speaker identification},
	  	author={Serizel, Romain and Bisot, Victor and Essid, Slim and Richard, Ga{\"e}l},
	  	booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
	  	pages={5},
	  	year={2017},
	  	organization={IEEE}
		}Download


References
++++++++++

.. [#] R. Serizel, S. Essid, and G. Richard. “Group nonnegative matrix factorisation with speaker and session variability compensation for speaker identification”. In Proc. of *2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 5470-5474, 2016.

.. [#] J. Mairal, F. Bach and J. Ponce. "Task-driven dictionary learning". In *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol 34(4), pp. 791--804, 2012.

.. [#] P. Sprechmann, A. M. Bronstein and G. Sapiro. "Supervised non-euclidean sparse NMF via bilevel optimization with applications to speech enhancement". In Proc. *Joint Workshop on Hands-free Speech Communication and Microphone Arrays (HSCMA)*, pp. 11-15, 2014.

.. [#] H. Zou and T. Hastie. "Regularization and variable selection via the elastic net". In *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, vol 67(2), pp. 301--320, 2015.

.. _GNMF: http://rserizel.github.io/groupNMF/intro.html
.. _similarity constraints: http://rserizel.github.io/groupNMF/intro.html#class-and-session-similarity-constraints