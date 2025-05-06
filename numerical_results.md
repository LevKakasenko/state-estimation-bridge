# Setup
The .mat files containing the turbulence data snapshots are located [here](https://zenodo.org/records/7464957). Load all these .mat files into the *2DTurbData/snapshots* folder, then run *2DTurbData_export.py* (located in the *2DTurbData* folder), which will generate a pickle file (*2DTurbData.pkl*) that can be read by our Python programs.

# Plot Generation
Below we provide instructions to generate each figure
in our *Numerical Results* section.  For descriptions of what each 
parameter represents, see the comments of *compute_error.py* in addition
to the comments in each of the Python files.  Many of our figures
require running *compute_error.py*, which at regular intervals saves the 
output of its computations into a file in the *logs* folder.  Each file contained
in the *logs* folder has a unique 8-digit ID, and contains information concerning the 
reconstruction error and sensor locations of a given combination of sensor placement
algorithm (Greedy D-Optimal, CPQR, etc.) and state estimation formula (DEIM or MAP) 
on a given dataset.  To get a summary of the files contained in the *logs* folder, 
run *log_summary.py*.  

Note that this code is intended only for prior covariance matrices that are diagonal and 
measurement noise covariance matrices that are a multiple of the identity.   

In the *Numerical Results* section of our paper, we discuss the greedy D-optimal and brute-force D-optimal sensor placement algorithms, and refrain from discussing A-optimal and greedy A-optimal sensor placement.  However, the performance of these two approaches can be seen by (1) running *compute_error.py* with the *heur* parameter set to 'aopt', (2) running *compute_error.py* with the *heur* parameter set to 'aopt_greedy', and (3) running *plot_error.py* with the *heuristics* parameter set to ['aopt', 'aopt_greedy'].

## Figure 2
Run *sensor_locations__Fourier.py* with the following parameter values:  

| Parameter         |               |
| ----------------- | ------------- |
| data_matrix       | 'fourier'     |     
| num_features_full | 10**3         |
| num_features      | 40            |
| num_samples       | 1000          |
| split_idx         | 750           |
| num_modes         | 20            |
| num_sensors       | 5             |
| noise             | .1            |
| sigma_noise       | .1            |
| heuristics        | ['opt_map', 'opt_deim', 'dopt', 'dopt_greedy', 'cpqr'] |
| seed              | 0             |


## Figure 3
Run *reconstruction_plot__Fourier.py* with the following parameter values:  

| Parameter         |               |
| ----------------- | ------------- |
| data_matrix       | 'fourier' |
| test_idx          | 10 |
| num_features_full | 10**3 |
| num_features      | 40 |
| num_samples       | 1000 |
| split_idx         | 750 |
| num_modes         | 20 |
| num_sensors       | 5 |
| noise             | .1 |
| sigma_noise       | .1 |
| heuristics        | ['dopt', 'dopt_greedy', 'cpqr'] |
| reconstructors    | ['map', 'map', 'deim'] |
| seed              | 0 |

In addition to plots of the reconstructions of a sample harmonic function, printed in the terminal are the relative
errors of these reconstructions (1) on this single sample and (2) averaged across all test samples.
Also printed in the terminal are the sensor locations as numerical values.

## Figure 4
To generate each of the plotted quantities (D-MAP, Greedy D-MAP, Q-DEIM) in Figure 4 (whose panel indices are included
in parentheses in the column headers below), first run *compute_error.py* with the following parameter values:  

| Parameters | D-MAP (a)  | D-MAP (b) | Greedy D-MAP (a/c) | Greedy D-MAP (b) | Greedy D-MAP (d)   | Q-DEIM (c) | Q-DEIM (d) |
| ---------- | ---------- | --------- | --------------- | ------------------- | ------------------ | ---------- | ---------- |
| data_matrix | 'fourier' | 'fourier' | 'fourier'       | 'fourier'           | 'fourier'          | 'fourier'  | 'fourier'  |
| num_features | 40       | 20        | 40              | 20                  | 40                 | 40         | 40         |
| num_samples  | 1000     | 1000      | 1000            | 1000                | 1000               | 1000       | 1000       |
| noise        | .1       | .1        | .1              | .1                  | .1                 | .1         | .1         |
| sigma_noise  | .1       | .1        | .1              | .1                  | .1                 | .1         | .1         |
| split_idx    | 750      | 750       | 750             | 750                 | 750                | 750        | 750        |
| mode_range   | range(1, 21) | range(20, 21) | range(1, 21) | range(20, 21)  | range(20, 21)      | range(1, 21) | range(20, 21) |
| sensor_range | range(5, 6)  | range(1, 11)  | range(5, 6)  | range(1, 11)   | range(1, 11)       | range(5, 6) | range(1, 11) |
| heur         | 'dopt'   | 'dopt'    | 'dopt_greedy'   | 'dopt_greedy'       | 'dopt_greedy'      | 'cpqr'     | 'cpqr_fair' |
| recons       | 'map'    | 'map'     | 'map'           | 'map'               | 'map'              | 'deim'     | 'deim'     |
| prior        | 'natural' | 'natural' | 'natural'      | 'natural'           | 'natural'          | 'identity' | 'identity' |
| seed         | 0        | 0         | 0               | 0                   | 0                  | 0          | 0          |

Then, to generate each of the panels in Figure 4, run *plot_error.py* with the following
parameter values:

| Parameters | Panel (a) | Panel (b) | Panel (c) | Panel (d) |
| ----------- | --------- | --------- | --------- | --------- |
| data_matrix | 'fourier' | 'fourier' | 'fourier' | 'fourier' |
| num_features | 40       | 20        | 40        | 40        |
| num_samples  | 1000     | 1000      | 1000      | 1000      |
| noise        | .1       | .1        | .1        | .1        |
| sigma_noise  | .1       | .1        | .1        | .1        |
| split_idx    | 750      | 750       | 750       | 750       |
| mode_range   | range(1, 21) | range(20, 21) | range(1, 21) | range(20, 21) |
| sensor_range | range(5, 6)  | range(1, 11) | range(5, 6) | range(1, 11) |
| plot_type    | 'error_vs_modes' | 'error_vs_sensors' | 'error_vs_modes' | 'error_vs_sensors' |
| heuristics   | ['dopt', 'dopt_greedy'] | ['dopt', 'dopt_greedy'] | ['dopt_greedy', 'cpqr'] | ['dopt_greedy', 'cpqr_fair'] |
| reconstructors | ['map', 'map']          | ['map', 'map']          | ['map', 'deim'] | ['map', 'deim'] | 
| priors         | ['natural', 'natural']  | ['natural', 'natural'] | ['natural', 'identity'] | ['natural', 'identity'] |
| seed            | 0     | 0         | 0         | 0         |


## Figures 5 & 9
To generate Figures 5 and 9 (whose panel indices are included in parentheses in the column headers below),
run *risk_prem_components_plot.py* with the following parameter values:

| Parameters | Figure 5 (a) |  Figure 5 (b) | Figure 9 (a) | Figure 9 (b) |
| ---------- | ------------ | ------------- | ------------ | ------------ |
| data_matrix | 'fourier'   | 'fourier'     | 'turbulence' | 'turbulence' |
| num_features | 40         | 40            | turbulence().shape[0] |turbulence().shape[0] |
| num_samples  | 1000       | 1000          | turbulence().shape[1] | turbulence().shape[1] |
| noise        | .1         | .1            | .3           | .3           |
| sigma_noise  | .1         | .1            | .3           | .3           |
| split_idx    | 750        | 750           | 750          | 750          |
| mode_range   | range(1, 21) | range(8, 21) | range(1, 101) | range(50, 101) |
| num_sensors | 5           | 5             | 25           | 25           |
| prior      | 'natural'    | 'natural'     | 'natural'    | 'natural'    |
| seed       | 0            | 0             | 0            | 0            |


## Figures 6 & 10
To generate Figures 6 and 10 (whose entries and panel indices are included in the column headers below),
first run *compute_error.py* with the following parameter values:

| Parameters | Greedy D-opt. - Figure 6 (a) | Q-MAP - Figure 6 (a) | Greedy D-opt. - Figure 6 (b) | Q-MAP - Figure 6 (b) | Greedy D-opt. - Figure 10 (a) | Q-MAP - Figure 10 (a) | Greedy D-opt. - Figure 10 (b) | Q-MAP - Figure 10 (b) |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| data_matrix  | 'fourier'    | 'fourier'    | 'fourier'    | 'fourier'    | 'turbulence' | 'turbulence' | 'turbulence' | 'turbulence' |
| num_features | 40           | 40           | 40           | 40           | turbulence().shape[0] | turbulence().shape[0] | turbulence().shape[0] | turbulence().shape[0] |
| num_samples  | 1000         | 1000         | 1000         | 1000         | turbulence().shape[1] | turbulence().shape[1] | turbulence().shape[1] | turbulence().shape[1] |
| noise        | .1           | .1           | .3           | .3           | .1           | .1           | .3           | .3           |
| sigma_noise  | .1           | .1           | .3           | .3           | .1           | .1           | .3           | .3           |
| split_idx    | 750          | 750          | 750          | 750          | 750          | 750          | 750          | 750          |
| mode_range   | range(1, 21) | range(1, 21) | range(1, 21) | range(1, 21) | range(5, 101, 5) | range(5, 101, 5) | range(5, 101, 5) | range(5, 101, 5) |
| sensor_range | range(5, 6)  | range(5, 6)  | range(5, 6)  | range(5, 6)  | range(25, 26) | range(25, 26) | range(25, 26) | range(25, 26) |
| heur         | 'dopt_greedy' | 'cpqr'      | 'dopt_greedy' | 'cpqr'      | 'dopt_greedy' | 'cpqr'      | 'dopt_greedy' | 'cpqr'      |
| recons       | 'map'        | 'map'        | 'map'        | 'map'        | 'map'        | 'map'        | 'map'        | 'map'        |
| prior        | 'natural'    | 'natural'    | 'natural'    | 'natural'    | 'natural'    | 'natural'    | 'natural'    | 'natural'    |
| seed         | 0            | 0            | 0            | 0            | 0            | 0            | 0            | 0            |


Then run *dice_plot.py* with the following parameter values:
| Parameters | Figure 6 (a) | Figure 6 (b) | Figure 10 (a) | Figure 10 (b) |
| ---------- | ------------ | ------------ | ------------- | ------------- |
| data_matrix | 'fourier'   | 'fourier'    | 'turbulence'  | 'turbulence'  |
| num_features | 40         | 40           | turbulence().shape[0] | turbulence().shape[0] |
| num_samples  | 1000       | 1000         | turbulence().shape[1] | turbulence().shape[1] |
| noise        | .1         | .3           | .1            | .3            |
| sigma_noise  | .1         | .3           | .1            | .3            |
| split_idx    | 750        | 750          | 750           | 750           |
| mode_range   | range(1, 21) | range(1, 21) | range(5, 101, 5) | range(5, 101, 5) |
| sensor_range | range(5, 6) | range(5, 6) | range(25, 26) | range(25, 26) |
| plot_type    | 'error_vs_modes' | 'error_vs_modes' | 'error_vs_modes' | 'error_vs_modes' |
| heuristics   | ['cpqr', 'dopt_greedy'] | ['cpqr', 'dopt_greedy'] | ['cpqr', 'dopt_greedy'] | ['cpqr', 'dopt_greedy'] |
| reconstructors | ['map', 'map'] | ['map', 'map'] | ['map', 'map'] | ['map', 'map'] |
| priors       | ['natural', 'natural'] | ['natural', 'natural'] | ['natural', 'natural'] | ['natural', 'natural'] |
| seed         | 0          | 0            | 0             | 0             |


## Figure 7
To generate each of the reconstructions in Figure 7 (given as column headers below), run 
*compute_error.py* with the following parameter values:

| Parameters   | Greedy D-MAP | Q-DEIM | Q-MAP  |
| ------------ | ------------ | ------ | ------ |
| data_matrix  | 'turbulence' | 'turbulence' | 'turbulence' |
| num_features | turbulence().shape[0] | turbulence().shape[0] | turbulence().shape[0] |
| num_samples  | turbulence().shape[1] | turbulence().shape[1] | turbulence().shape[1] |
| noise        | .3           | .3     | .3     |
| sigma_noise  | .3           | .3     | .3     |
| split_idx    | 750          | 750    | 750    |
| mode_range   | range(100, 101) | range(50, 51) | range(100, 101) |
| sensor_range | range(50, 51) | range(50, 51) | range(50, 51) |
| heur         | 'dopt_greedy' | 'cpqr_fair' | 'cpqr' |
| recons       | 'map'        | 'deim' | 'map'  |
| prior        | 'natural'    | 'identity' | 'natural' |
| seed         | 0            | 0      | 0      |

Then run *reconstruction_plot__turbulence.py* with the following parameter values:

| Parameters     | |
| -------------- | ------------ |
| data_matrix    | 'turbulence' |
| num_features   | turbulence().shape[0] |
| num_samples    | turbulence().shape[1] |
| noise          | .3 |
| sigma_noise    | .3 |
| split_idx      | 750 |
| num_modes      | 100 |
| num_sensors    | 50  |
| plot_type      | 'error_vs_sensors' |
| heuristics     | ['dopt_greedy', 'cpqr_fair', 'cpqr'] |
| reconstructors | ['map', 'deim', 'map'] |
| priors         | ['natural', 'identity', 'natural'] |
| seed           | 0 |
| sample         | 125 |

In addition to plots of the reconstructions of this sample snapshot, printed in the terminal are the relative errors of the reconstructions of this snapshot.

## Figure 8
To generate each of the plotted quantities (D-MAP, Greedy D-MAP, Q-DEIM) in Figure 8 (whose panel indices are included
in parentheses in the column headers below), first run *compute_error.py* with the following parameter values:  

| Parameters   | Greedy D-MAP (a/c) | Q-DEIM (a) | Greedy D-MAP (b/d) | Q-DEIM (b) | Q-MAP (c) | Q-MAP (d) |
| ------------ | ---------- | --------- | ------------------ | ---------------- | ------------------ | ---------- |
| data_matrix  | 'turbulence' | 'turbulence' | 'turbulence' | 'turbulence' | 'turbulence' | 'turbulence' |
| num_features | turbulence().shape[0] | turbulence().shape[0] | turbulence().shape[0] | turbulence().shape[0] | turbulence().shape[0] | turbulence().shape[0] |
| num_samples  | turbulence().shape[1] | turbulence().shape[1] | turbulence().shape[1] | turbulence().shape[1] | turbulence().shape[1] | turbulence().shape[1] |
| noise        | .3         | .3        | .3                  | .3              | .3                 | .3         |
| sigma_noise  | .3         | .3        | .3                  | .3              | .3                 | .3         |
| split_idx    | 750        | 750       | 750                 | 750             | 750                | 750        |
| mode_range   | range(5, 101, 5) | range(5, 101, 5) | range(100, 101)  | range(100, 101) | range(5, 101, 5) | range(100, 101) |
| sensor_range | range(25, 26)    | range(25, 26)    | range(5, 101, 5) | range(5, 101, 5) | range(25, 26)   | range(5, 101, 5) |
| heur         | 'dopt_greedy'    | 'cpqr'           | 'dopt_greedy'    | 'cpqr_fair'     | 'cpqr'    | 'cpqr'    |
| recons       | 'map'      | 'deim'                 | 'map'            | 'deim'          | 'map'     | 'map'     |
| prior        | 'natural'  | 'identity'             | 'natural'        | 'identity'      | 'natural' | 'natural' |
| seed         | 0 | 0 | 0 | 0 | 0 | 0 |

Then, to generate each of the panels in Figure 8, run *plot_error.py* with the following
parameter values:

| Parameters | Panel (a) | Panel (b) | Panel (c) | Panel (d) |
| ----------- | --------- | --------- | --------- | --------- |
| data_matrix | 'turbulence' | 'turbulence' | 'turbulence' | 'turbulence' |
| num_features | turbulence().shape[0] | turbulence().shape[0] | turbulence().shape[0] | turbulence().shape[0] |
| num_samples  | turbulence().shape[1] | turbulence().shape[1] | turbulence().shape[1] | turbulence().shape[1] |
| noise        | .3       | .3       | .3       | .3       |
| sigma_noise  | .3       | .3       | .3       | .3       |
| split_idx    | 750      | 750      | 750      | 750      |
| mode_range   | range(5, 101, 5) | range(100, 101) | range(5, 101, 5) | range(100, 101) |
| sensor_range | range(25, 26) | range(5, 101, 5) | range(25, 26) | range(5, 101, 5) |
| plot_type    | 'error_vs_modes' | 'error_vs_sensors' | 'error_vs_modes' | 'error_vs_sensors' |
| heuristics   | ['dopt_greedy', 'cpqr'] | ['dopt_greedy', 'cpqr_fair'] | ['dopt_greedy', 'cpqr'] | ['dopt_greedy', 'cpqr'] |
| reconstructors | ['map', 'deim'] | ['map', 'deim'] | ['map', 'map'] | ['map', 'map'] |
| priors         | ['natural', 'identity'] | ['natural', 'identity'] | ['natural', 'natural'] | ['natural', 'natural'] |
| seed         | 0 | 0 | 0 | 0 |



# Other Numerical Results (not plotted in the article)
## Singular value decay
To observe the singular value decay of our datasets, run *singular_vals_plot.py* with the following
parameter values:

| Parameters   | Fourier Data | Turbulence Data |
| ------------ | ------------ | --------------- |
| data_matrix  | 'fourier'    | 'turbulence'    |
| num_features | 40 | turbulence().shape[0] |
| num_samples  | 1000 | turbulence().shape[1] |
| split_idx    | 750 | 750 |
| seed         | 0 | 0 |


## Noise percentages
In our numerical results, we report that measurement noise represents approximately 14.5% of a Fourier test data sample
and 15% of a turbulence test data sample.  We compute this quantity as the relative error of a noisy 
data sample with respect to the true data sample, averaged over all test data samples.  To compute these
quantities, run *noise_error.py* with the following parameter values:

| Parameters   | Fourier Noise Error | Turbulence Noise Error |
| ------------ | ------------------- | ------------------- |
| data_matrix  | 'fourier' | 'turbulence' |
| num_features | 40 | turbulence().shape[0] |
| num_samples  | 1000 | turbulence().shape[1] |
| noise        | .1 | .3 |
| split_idx    | 750 | 750 |
| seed         | 0 | 0 |

## Explained variance in the turbulence data
To see the total portion of training data variance explained by the first *n* POD modes of a dataset,
and to see the minimum number of POD modes needed to explain *thres* of the training data variance (in our example, *thres* is set to .95, meaning we seek to explain 95% of training data variance),
run *explained_variance_plot.py* with the following parameter values:

| Parameters   | Fourier Data | Turbulence Data |
| ------------ | ------------ | --------------- |
| data_matrix  | 'fourier'    | 'turbulence'    |
| num_features | 40 | turbulence().shape[0] |
| num_samples  | 1000 | turbulence().shape[1] |
| split_idx    | 750 | 750 |
| thres | .95 | .95 |
| seed | 0 | 0 |

Displayed in the terminal are the minimum number of modes needed to explain *thres* of the 
training data variance, where *thres* is some threshold between 0 and 1.
In the case of the turbulence data, also displayed is the total variance 
explained by the first 100 POD modes.

## Dip in the smallest singular value of S^T Phi
In Section 5.1.1, we remark that the smallest singular value of S^T Phi dips close to zero 
precisely when the number of modes equals the number of sensors.  To see this dip, run
*smallest_singular_val_plot.py* with the following parameter values:

| Parameters   | Fourier Data | Turbulence Data |
| ------------ | ------------ | --------------- |
| data_matrix  | 'fourier'    | 'turbulence'    |
| num_features | 40 | turbulence().shape[0] |
| num_samples  | 1000 | turbulence().shape[1] |
| split_idx    | 750 | 750 |
| mode_range | range(1, 11) | range(1, 51) |
| num_sensors | 5 | 25 |
| seed | 0 | 0 |
