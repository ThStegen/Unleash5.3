# Unleash project WP5.3 T5.2 Availability of the model for integrating the VPP into distribution grids

Simulator of the impact of flexibility usage on distribution grids.

## Part 1: Implicit flexibility

The first module ``OperationalPlanning.py`` models the optimal reaction of users to the tariffs they are subject to.
By planning battery, electric vehicl, and water boiler operation on a day-ahead horizon, users can maximize photovoltaic usage and minimize remaining energy bill.

## Part 2: Explicit flexibility

The second part, in ``ExplicitFlex.py``, models an aggregated automatic reaction of the water boilers to external signals for external flexibility needs.
The reaction, depending on the remaining flexibility, is both computed and automatically automatically, to react to FCR and aFRR market signals.

## Part 3: Power flow analysis

The file ``PowerFlow.py`` contains the function for _a posteriori_ power flow analysis.
It uses pandapower and its included test networks to see the effect of each scenario on the grid performance. 

## Additional files

- inputs management
- case definition
- tariff design
