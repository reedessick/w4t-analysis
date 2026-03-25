This page summarizes the results of running our structure-function-to-She-Leveque inference process for the (10k)^3 simulation data that James had on hand.
We use the transverse and longitudinal structure functions he had previously calculated for O(20) snapshots.

We estimate measurement uncertainties with the sample variance of each increment across snapshots.
To help with numerical efficiency within the sampling proceedure, we additionally

  * set a lower limit on the relative uncertainty in each measured structure function of > 5%. This primarily affects low-order structure functions
  * resample the measured structure function to a logarithmic grid (instead of linear) and only retain 51 sample points. We linearly interpolate both the mean and standard deviation of the measured structure function.

Below, I show the inferred structure functions as a function of scale along with the She-Leveque parameters and violin plots.
These are separated by field (either `vel` or `mag`) and direction (`long` or `trsv`).

The She-Leveque ansatz is taken to be a prediction for the averaged scaling exponent over some range of scales.
I consider several different ranges (with boundaries chosen by eye to correspond to approximately constant logarithmic derivatives).
Results for each range are shown separately as well.

---

# vel

---

## vel trsv

### vel trsv 3e-4 6e-4

|structure function|scaling exponent|
|---|---|
|<img src="fits/vel-trsv/plot-structure-function-ansatz_vel-trsv-3e-4_6e-4.png">|<img src="fits/vel-trsv/plot-scaling-exponent-ansatz_vel-trsv-3e-4_6e-4.png">|
|<img src="fits/vel-trsv/plot-structure-function-ansatz-logarithmic-derivative_vel-trsv-3e-4_6e-4.png">|<img src="fits/vel-trsv/w4t-corner_vel-trsv-3e-4_6e-4.png">|

### vel trsv 3e-4 6e-4

|structure function|scaling exponent|
|---|---|
|<img src="fits/vel-trsv/plot-structure-function-ansatz_vel-trsv-4e-3_8e-3.png">|<img src="fits/vel-trsv/plot-scaling-exponent-ansatz_vel-trsv-4e-3_8e-3.png">|
|<img src="fits/vel-trsv/plot-structure-function-ansatz-logarithmic-derivative_vel-trsv-4e-3_8e-3.png">|<img src="fits/vel-trsv/w4t-corner_vel-trsv-4e-3_8e-3.png">|

### vel trsv 3e-4 6e-4

|structure function|scaling exponent|
|---|---|
|<img src="fits/vel-trsv/plot-structure-function-ansatz_vel-trsv-7e-2_14e-2.png">|<img src="fits/vel-trsv/plot-scaling-exponent-ansatz_vel-trsv-7e-2_14e-2.png">|
|<img src="fits/vel-trsv/plot-structure-function-ansatz-logarithmic-derivative_vel-trsv-7e-2_14e-2.png">|<img src="fits/vel-trsv/w4t-corner_vel-trsv-7e-2_14e-2.png">|

---

## vel long

### vel long 3e-4 6e-4

|structure function|scaling exponent|
|---|---|
|<img src="fits/vel-long/plot-structure-function-ansatz_vel-long-3e-4_6e-4.png">|<img src="fits/vel-long/plot-scaling-exponent-ansatz_vel-long-3e-4_6e-4.png">|
|<img src="fits/vel-long/plot-structure-function-ansatz-logarithmic-derivative_vel-long-3e-4_6e-4.png">|<img src="fits/vel-long/w4t-corner_vel-long-3e-4_6e-4.png">|

### vel long 7e-3 14e-3

|structure function|scaling exponent|
|---|---|
|<img src="fits/vel-long/plot-structure-function-ansatz_vel-long-7e-3_14e-3.png">|<img src="fits/vel-long/plot-scaling-exponent-ansatz_vel-long-7e-3_14e-3.png">|
|<img src="fits/vel-long/plot-structure-function-ansatz-logarithmic-derivative_vel-long-7e-3_14e-3.png">|<img src="fits/vel-long/w4t-corner_vel-long-7e-3_14e-3.png">|

### vel long 6e-2 12e-2

|structure function|scaling exponent|
|---|---|
|<img src="fits/vel-long/plot-structure-function-ansatz_vel-long-6e-2_12e-2.png">|<img src="fits/vel-long/plot-scaling-exponent-ansatz_vel-long-6e-2_12e-2.png">|
|<img src="fits/vel-long/plot-structure-function-ansatz-logarithmic-derivative_vel-long-6e-2_12e-2.png">|<img src="fits/vel-long/w4t-corner_vel-long-6e-2_12e-2.png">|

---

# mag

---

## mag trsv

### mag trsv 7e-4 14e-4

|structure function|scaling exponent|
|---|---|
|<img src="fits/mag-trsv/plot-structure-function-ansatz_mag-trsv-7e-4_14e-4.png">|<img src="fits/mag-trsv/plot-scaling-exponent-ansatz_mag-trsv-7e-4_14e-4.png">|
|<img src="fits/mag-trsv/plot-structure-function-ansatz-logarithmic-derivative_mag-trsv-7e-4_14e-4.png">|<img src="fits/mag-trsv/w4t-corner_mag-trsv-7e-4_14e-4.png">|

---

## mag long

### mag long 7e-4 14e-4

|structure function|scaling exponent|
|---|---|
|<img src="fits/mag-long/plot-structure-function-ansatz_mag-long-7e-4_14e-4.png">|<img src="fits/mag-long/plot-scaling-exponent-ansatz_mag-long-7e-4_14e-4.png">|
|<img src="fits/mag-long/plot-structure-function-ansatz-logarithmic-derivative_mag-long-7e-4_14e-4.png">|<img src="fits/mag-long/w4t-corner_mag-long-7e-4_14e-4.png">|
