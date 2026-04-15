<p align="center">
  <img src="./1734935629227.jpeg">
</p>

# Amenities Magnet

Amenities Magnet is a Streamlit application for exploring rental price patterns across German housing markets.
It combines market exploration, regional comparisons, and model-oriented views built from cleaned and transformed rental listing data.

## Overview

The project focuses on understanding which housing attributes are most associated with rental price per square meter.
The repository includes:

- a Streamlit dashboard for interactive exploration
- cleaned and transformed datasets for analysis and modeling
- notebooks, training scripts, and model artifacts used for experimentation

## Features

- State and city market exploration
- Modeling-oriented feature views
- Regional summaries and comparisons
- Benchmarking based on comparable property profiles

## Repository Structure

- `Code/` Streamlit application code
- `Data/` cleaned and transformed datasets
- `Analysis/` notebooks, training scripts, and artifacts
- `Visualizations/` exported charts and supporting visuals

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run Code/app.py
```

## Deployment

This project can be deployed on Streamlit Community Cloud using `Code/app.py` as the entrypoint.

## Notes

- Large dataset files are tracked with Git LFS.
- If a CSV appears as a small pointer file, run `git lfs pull` inside the repository.
