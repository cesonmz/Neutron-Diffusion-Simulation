🔬 Neutron Diffusion Simulation

This project explores the neutron diffusion equation using both numerical methods and reactor physics concepts. It focuses on understanding how neutron flux behaves in different geometries and how system size affects criticality.

The goal is not just to run simulations, but to understand the physics and mathematics behind them.

🚀 Overview

This repository implements:

✅ 1D steady-state neutron diffusion (eigenvalue problem)
✅ Analytical vs numerical flux comparison
✅ Criticality analysis (k_eff vs system size)
✅ 2D neutron flux distribution
✅ Core + reflector modeling
✅ Bare vs reflected reactor comparison

All simulations are based on one-group diffusion theory and solved using finite difference methods.

📊 Results
1D Flux (Numerical vs Analytical)
Flux follows cosine shape
Strong agreement with theory
Example: k_eff ≈ 2.23 for large slab
Criticality Sweep
Shows how geometry affects reactor behavior
Critical thickness ≈ 9–10 cm
Larger systems → less leakage → higher k_eff
2D Bare Core
Symmetric flux distribution
Peak at center
k_eff ≈ 1.99
Matches diffusion theory predictions
2D Core + Reflector
Flux extends into reflector
Reduced leakage
Higher k_eff compared to bare system

🎯 Motivation

This project was created to:

Move beyond black-box simulation tools
Understand reactor physics from first principles
Apply linear algebra + differential equations to real systems
Explore deterministic methods alongside Monte Carlo (e.g., OpenMC)

🤖 AI Acknowledgement

AI tools (ChatGPT) were used to assist with:

Code debugging; Numerical method clarification; Report structuring

All physics understanding and final implementation were verified independently.
