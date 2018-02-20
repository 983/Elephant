Based on [this paper](https://fermatslibrary.com/s/drawing-an-elephant-with-four-complex-parameters).

Elephant from paper with just 8 parameters:

![Image of elephant plot](https://raw.githubusercontent.com/983/Elephant/master/elephant_plot.png)

Fancy elephant with more parameters:

![Image of fancy elephant plot](https://raw.githubusercontent.com/983/Elephant/master/fancy_elephant_plot.png)

# How does it work?

Given arrays of parameters `A_k^x, B_k^x, A_k^y, B_k^y`, a curve, parameterized by `t`, is calculated and the Adam optimizer is used to fit the curve to given data points `p_i` describing the contour of an elephant:

```
  x(t) = \sum_{k=0} (A_k^x cos(kt) + B_k^x sin(kt))
  y(t) = \sum_{k=0} (A_k^y cos(kt) + B_k^y sin(kt))
  
  min_{A_k^x, B_k^x, A_k^y, B_k^y} \sum_i ((x(t) - p_i^x)^2 + (y(t) - p_i^y)^2)
```
