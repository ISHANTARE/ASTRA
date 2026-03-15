# ASTRA (Autonomous Space Traffic Risk Analyzer) - Project Overview

## 1. Introduction
ASTRA is a Space Situational Awareness (SSA) analysis platform designed to visualize satellites and space debris orbiting Earth, analyze orbital congestion, detect close approaches between space objects, and generate prediction reports for the next 24 hours.

## 2. Core Objectives
Build a web platform that:
- Loads all tracked orbital objects (~30,000+ objects).
- Visualizes them in real time around Earth.
- Predicts close approaches between objects using the SGP4 propagation model.
- Analyzes orbital congestion by altitude bands.
- Generates research-level insights and prediction reports.

## 3. Scope & Scale
The platform must be designed so that it can evolve into a research project and academic thesis system. 
- **Initial Dataset:** ~30,000 objects (Active Satellites ~9k, Debris Fragments ~20k, Rocket Bodies ~2k).
- **Prediction Window:** 24 hours.
- **Simulation Resolution:** 5 minutes.
- **Total Simulation Steps:** 288 steps per object pair analyzed.

## 4. Inspirations & References
- **Visualization Inspiration:** *Stuff in Space*
- **Operational Inspiration:** Simplified SSA systems similar to those used by organizations like **NASA** and the **European Space Agency (ESA)**.

## 5. Restrictions and Requirements
- All computation must run **server-side**.
- Heavy analysis computation must **only run after user filtering**.
- **Never compute pairwise comparisons across all 30k objects** simultaneously.
- Always implement the multi-stage filtering algorithm.
- Ensure visualization remains smooth even with 30,000 objects rendered.
