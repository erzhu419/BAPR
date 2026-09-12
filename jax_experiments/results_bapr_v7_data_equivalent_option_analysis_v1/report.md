# BAPR-v7 unique-data-equivalent option development screen

Oracle capacity gate: **FAIL**  
Learned persistent gate: **FAIL**

| Event | Source | Stationary | Slow pair | Full cycle | Full - hard |
|---:|---|---:|---:|---:|---:|
| 6100 | hard CUSUM | 2673.6 | 2620.1 | 2709.3 | +0.0 |
| 6100 | v7 robust | 2130.0 | 2217.6 | 2065.0 | -644.4 |
| 6100 | v7 oracle option | 2716.7 | 2894.1 | 2654.2 | -55.1 |
| 6100 | v7 learned option | 2720.0 | 2622.7 | 2209.5 | -499.9 |
| 6200 | hard CUSUM | 2615.7 | 2828.0 | 2785.4 | +0.0 |
| 6200 | v7 robust | 2129.9 | 2126.9 | 2079.3 | -706.1 |
| 6200 | v7 oracle option | 2716.9 | 2724.6 | 2804.8 | +19.4 |
| 6200 | v7 learned option | 2703.5 | 2626.2 | 2385.6 | -399.8 |

The oracle gate is evaluated first. A failed oracle gate closes this persistent-option architecture; a passed oracle gate with a failed learned gate localizes the remaining problem to inference or option boundary selection.
