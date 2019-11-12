1. In test_train_split.txt, the first line is the percentage of
   historical trajectories used for training, the second is the
   percentage used for validation (for the GMM), and the third is
   the percentage used for testing. For the transition matrix, both
   training and validation trajectories are used for training.
2. In settings.txt, the first line is the number of KCs, and the
   second is the number of proficiency levels.

In order to run the Python scripts to see the performance of the
sim2 baseline, do the following:
1. First, prepare test_train_split.txt and settings.txt. Also, place
   MDPdatasheet.csv and action.csv in this directory.
2. Then, run initial_proficiency_gmm.py. This trains a GMM on a
   subset of the historical trajectories to find the distribution
   of the initial proficiencies, and also shows the performance of
   the GMM on the remaining trajectories (evaluated using similarity
   of the distributions/KL divergence, which is equivalent to likelihood).
3. Then, run parseDatasheet.py. This creates a transition matrix
   estimated from the same historical trajectories as those used
   for the GMM.
4. Then, run test_second_simulator.py, which shows the average log-likelihood
   of each transition according to the transition matrix (conditional on the 
   history up to that point, represented by a belief state at each time step).

I will make a makefile soon.