This folder is where the log files outputted by *compute_error.py* are
stored.  Each log file has a unique 8-digit ID.  

Each log file contains metadata concerning 
* The dataset used for training and testing (either the Fourier or turbulence dataset).
* The number of features per data sample.
* The number of data samples used.
* The number of training and testing samples (specified by a column index that splits the array of data).
* The type of sensor placement algorithm used (CPQR, D-optimal, Greedy D-optimal, A-optimal, or Greedy A-optimal).
* The full-state reconstructor (DEIM or MAP).
* The type of prior covariance matrix used (either that which we propose in Section 3.1 or the identity).
* The standard deviation of the uncorrelated Gaussian noise added to the test data.
* The model parameter for a measurement noise standard deviation (multiplying this parameter by the identity gives the Gamma_noise covariance matrix).
* The random seed used.  

In addition, each log file contains key/value pairs.  Each key is a tuple containing a given number of modes
and sensors.  Each value is a tuple of the corresponding sensor locations, reconstruction error mean and reconstruction error variance (on the test data) computed for the given number of modes and sensors.  

All log files can be summarized by running *log_summary.py*.
