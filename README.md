# Bridging the Gap Between Deterministic and Probabilistic Approaches to State Estimation
## Overview 
The code in this repository complements the paper "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025). We originally ran this code in Python 3.9.15.  To reproduce our numerical results, follow the instructions in *numerical_results.md*.  

## Work used in this repository
As part of this repository, we implement the DEIM formula first proposed by Chaturantabut and Sorensen (2010) and subsequently repurposed by Manohar et al. (2018), in addition to the Column-Pivoted QR sensor placement algorithm by Drmač and Gugercin (2016).  We also implement a greedy algorithm whose performance guarantees were first proven by Nemhauser et al. (1978), and which was later analyzed and implemented in the context of D-optimal sensor placement by Krause et al. (2008), Shamaiah et al. (2010), and Nishida et al. (2022).  The speed of the greedy D-optimal algorithm that we implement herein can be further improved by implementing the rank-1 inverse update described in Algorithm 2 of Shamaiah et al. (2010) and in Section 4.A. of Nishida et al. (2022).  We also use the Dice coefficient as defined in Dice (1945).

## References to work used in this repository
S. Chaturantabut, D. C. Sorensen, Nonlinear model reduction via discrete empirical interpolation, SIAM Journal on Scientific Computing 32 (5)
(2010) 2737–2764. https://doi.org/10.1137/090766498  

K. Manohar, B. W. Brunton, J. N. Kutz, S. L. Brunton, Data-driven sparse
sensor placement for reconstruction: Demonstrating the benefits of exploiting known patterns, IEEE Control Systems Magazine 38 (3) (2018) 63–86.
https://doi.org/10.1109/MCS.2018.2810460    

Z. Drmač, S. Gugercin, A new selection operator for the discrete empirical interpolation method—improved a priori error bound and extensions, SIAM Journal on Scientific Computing 38 (2) (2016) A631–A648.
https://doi.org/10.1137/15M1019271

G. L. Nemhauser, L. A. Wolsey, M. L. Fisher, An analysis of approximations
for maximizing submodular set functions—I, Mathematical Programming
14 (1) (1978) 265–294. https://doi.org/10.1007/BF01588971  

A. Krause, A. Singh, C. Guestrin, Near-optimal sensor placements in gaussian processes: Theory, efficient algorithms and empirical studies, J. Mach.
Learn. Res. 9 (2008) 235–284.
http://jmlr.org/papers/v9/krause08a.html  

M. Shamaiah, S. Banerjee, H. Vikalo, Greedy sensor selection: Leveraging
submodularity, in: 49th IEEE Conference on Decision and Control (CDC),
2010, pp. 2572–2577. https://doi.org/10.1109/CDC.2010.5717225  

T. Nishida, N. Ueno, S. Koyama, H. Saruwatari, Region-restricted sensor placement based on gaussian process for sound field estimation, IEEE
Transactions on Signal Processing 70 (2022) 1718–1733. https://doi.org/10.1109/TSP.2022.3156012  

L. R. Dice, Measures of the amount of ecologic association between species,
Ecology 26 (3) (1945) 297–302. https://doi.org/10.2307/1932409  

## License
If you use the code in this repository in any form, see the [license](https://github.com/LevKakasenko/state-estimation-bridge/blob/main/LICENSE).  Also, please cite
"Bridging the Gap Between Deterministic and Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian, Mohammad Farazmand, and Arvind Krishna Saibaba (2025).
