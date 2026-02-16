'''
There's a third bias check that belongs in the model pipeline phase (Phase 3), not the data pipeline: 
testing whether the Green/Yellow/Red recommendations are fair across user demographics.
 For example, does the deterministic engine systematically give Red lights to low-income users 
 for reasonably affordable products? That's the bias check that actually matters for SavVio's mission,
and it can only be done once the decision engine exists.
'''