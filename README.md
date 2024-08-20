## QFpy v2
QFpy is a Python package implementing common numerical tools used in quantitative finance to estimate the value of financial instruments and the associated risk.

## Installation
Clone this repository to your local machine:

```bash
    git clone https://github.com/yourusername/quant-finance-lib.git
```

And locally install it with `pip`

```bash
    cd QFpy
    pip install -e ./
```

## Quickstart guide

The library is under development. So far, it contains analytic formula for the 
value and greeks of european call and put options (found in `QFpy.vanilla_options_utils`)
as well as a class to solve the Black-Scholes equation with implicit finite difference methods
(found in `QFpy.black_scholes`) and methods to evaluate the VaR of a portfolio of equity options 
or interest rate derivatives (cashflow-mappable) using either MonteCarlo techniques or a linear 
or quadratic approximation (in which case a Cornish-Fisher asymptotic expansion is used to compute 
the quantile including skewness effects, the effect of speed on the distribution is work in progress). The `Qpy.capm_utils` module contains a function that finds the efficient frontier (minimum risk for a given return) for a set of assets given their returns, volatilities and the correlations of their returns.
Examples can be found in the example directory, which also contains a blueprint for what will become a module of utilities for HJM modelling of forward rate curves in the Musiela formalism.

## Running the Unit Tests

QFpy supports the `tox` testing framework. If you already installed the package,
you should have all the necessary ingredients. All you need to do to run the tests is

```bash
    cd QFpy
    tox
```
This tends to be slow the first time because a fresh environment will be installed 
with all the dependencies. If you want to avoid this you can also manually run the tests
as scripts inside the `test` directory.

## Disclaimer

This library is developed as a learning tool for quantitative finance concepts. It is not intended for professional or commercial use. The authors make no warranties about the suitability or accuracy of this software for any purpose. Use at your own risk.
