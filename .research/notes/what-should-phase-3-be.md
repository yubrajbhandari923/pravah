---
title: What should Phase 3 be?
pinned: false
links: []
created: '2026-10-10T23:14:43-04:00'
---

# What should Phase 3 be?
There are a lot of things to be done, improving numerical techniques, better models, realistic data and so on.

But, before going too deep in technical need to make sure at this stage that our framework is translateble to new place, easily. Which means, key equations, its constraints and techniques used to solve are easy to move around/change (not a super big deal if not but still should not be too focused on Belauri's case).

Other than that, we want to encorporate population density data into out model, with the key motivation that when we have to give up inference precision (or like predictive power), we would rather give up for non-populous areas within our area of concern than treating every point as equally important. This fact will be important in ou Experimental design/Optimal Sensor Depolyment type question. We want a data layer for population information setup. 

And, we also want to include geographical information somehow. For e.g Geographical elevation does not influence the concentration field probably (if it does we will explore it later), but for initial guess we can consider that wind flow matters for flow of concentration and time evolution of wind flow is depedent on geography (like wind if blowing against a mountain, not within our region of interest, i.e Belauri, but 50 KM away from that, might eventually return back to Belauri with some time delay.)


These two experiments/ questions also seem important to be addressed in the next phase.

Priority 1 — Is the model numerically accurate enough to support sensor placement?
I would create an experiment specifically about the numerical sensitivity of scientific conclusions.
Use identical source configurations, wind fields, candidate locations, noise realizations, and evaluation scenarios at several grid resolutions. Change only the discretization.
Measure how source fingerprints, reconstructed concentrations, source-amplitude estimates, and optimal sensor configurations change with refinement.
The central question is:
Do the conclusions converge as the numerical resolution improves?
If the supposedly optimal sensor locations keep moving as the mesh is refined, they are not yet reliable deployment recommendations.
I would also compare the current transport discretization with a less diffusive numerical scheme, while checking conservation, positivity, and temporal convergence. Merely increasing spatial resolution without controlling time discretization can obscure the result.
Priority 2 — How accurately must we know the wind?
This is the experiment I find most scientifically promising.
Instead of testing only one wind perturbation, measure a controlled error-response relationship.
For example, hold the same synthetic truth cases fixed and vary the wind-direction error:
Proposed experiment: reconstruction error versus wind error

For each wind error, compare three inversion strategies:
- Known wind: Give the estimator the correct wind, providing an idealized reference.
- Incorrect fixed wind: Use the nominal wind and fit only emission rates, as PRAVAH currently does.
- Uncertain wind: Allow the estimator to account for a physically constrained range of wind conditions, while fitting emissions.
Then evaluate not only total concentration RMSE but also source-amplitude recovery, local/external attribution, and performance at sensors held out from fitting.
This could answer a genuinely useful question:
How does meteorological uncertainty limit our ability to infer pollution sources from sparse observations?
It could also lead naturally to uncertainty-aware sensor placement. Rather than choosing sites optimal for one assumed wind field, we could optimize expected performance over a distribution of plausible winds.
That is a stronger research direction than simply comparing additional greedy heuristics