# Structure Functions

Below, we summarize the scaling exponents (logarithmic derivatives) of structure functions averaged over many snapshots for each set of flow parameters and field.
Violin plots are the posterior distributions of the logarithmic derivative of our anstaz for the structure function (`S`) at scale `tau` of order `p`

```math
    \frac{d\log S^p_\tau}{d\log \tau} = \xi  - b_l \left( 1 + \left(\frac{s_l}{\tau}\right)^{-n_l} \right)^{-1} + b_h \left( 1 + \left(\frac{\tau}{s_h}\right)^{-n_h} \right)^{-1}
```

*Importantly*, fits are conducted in 2 ways.
The left-hand side of the violins (unfilled) represent separate fits for the scaling exponent at each order `p`.
The right-hand side of the vilins (filled) represent a simultaneous fit for all scaling exponents for all orders `p` based on the She-Leveque ansatz
```math
  \frac{d\log S^p_\tau}{d\log\tau} = \left(\frac{p}{3}\right)(1-x) + C_0 \left(1-\beta^{p/3}\right)
```
which is based on Eq 7.64 of "Magnetohydrodynamic Turbulence" (Dieter Biskamp).
 
Structure functions are color-coded by their order as follows

  * 1st order : blue
  * 2nd order : orange
  * 3rd order : green
  * 4th order : red
  * 5th order : purple
  * 6th order : brown

Flow parameters are represented in the rows and columns of each table.
Rows correspond to M and columns correspond to MA.

If you click on any image, you should be redirected to the PNG itself and can zoom in for more detail.

---

Below, I only show results for the "direct" structure function calculations.
These results are for the scaling exponent extracted at a reference scale of `32`.

## Density

### Violins for S vs p

|M/MA| 01 | 05 | 1 | 2 | 4 | 6 | 8 | 10 |
|----|----|----|---|---|---|---|---|----|
| 05 |<img src="M05MA01/w4t-plot-scaling-exponent-ansatz-violin-032_M05MA01_avrg_dens_dsf_32.png">
| 2  |
| 4  |
| 10 |

### Structure functions inferred assuming She-Leveque ansatz

|M/MA| 01 | 05 | 1 | 2 | 4 | 6 | 8 | 10 |
|----|----|----|---|---|---|---|---|----|
| 05 |<img src="M05MA01/w4t-plot-scaling-exponent-ansatz_M05MA01_avrg_dens_dsf_32.png">
| 2  |
| 4  |
| 10 |

## Velocity

### Direct Structure Function

|M/MA| 01 | 05 | 1 | 2 | 4 | 6 | 8 | 10 |
|----|----|----|---|---|---|---|---|----|
| 05 |
| 2  |
| 4  |
| 10 |

## Vorticity

### Direct Structure Function

|M/MA| 01 | 05 | 1 | 2 | 4 | 6 | 8 | 10 |
|----|----|----|---|---|---|---|---|----|
| 05 |
| 2  |
| 4  |
| 10 |

## Magnetic Field

### Direct Structure Function

|M/MA| 01 | 05 | 1 | 2 | 4 | 6 | 8 | 10 |
|----|----|----|---|---|---|---|---|----|
| 05 |
| 2  |
| 4  |
| 10 |

## Current Density

### Direct Structure Function

|M/MA| 01 | 05 | 1 | 2 | 4 | 6 | 8 | 10 |
|----|----|----|---|---|---|---|---|----|
| 05 |
| 2  |
| 4  |
| 10 |
