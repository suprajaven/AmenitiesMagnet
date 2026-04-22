 brew install git-lfs<p align="center">
  <img src="./1734935629227.jpeg"> 
</p>


# ***CSE6242: Data & Visual Analytics | Spring 2026 | Team 146***
 
This is Team 146's group project GitHub repository.
 
## ***Project Title:*** 
Amenities Magnet: Identifying Key Drivers of Residential Rental Prices in Germany.

## ***Overview*** 

Amenities Magnet is an interactive data visualization tool that predicts residential rental prices across German cities and explains *why* a property costs what it does — not just what it costs.

Built on ImmoScout24 listings data, the project trains city-stratified XGBoost and Random Forest models to capture nonlinear relationships between property features and rent. SHAP values are used to surface per-feature attributions, making the model's reasoning transparent and explorable. Results are presented through a Streamlit dashboard where users can compare feature importance rankings across cities.

### ***Key Features*** 

- **City-specific models** that reveal how rent drivers shift across German markets
- **SHAP-powered explainability** showing which amenities push prices up or down
- **Multi-model benchmarking** (XGBoost vs. Random Forest vs. baseline)
- **Interactive Streamlit dashboard** for exploring feature importance by city

### ***Who Is This For?*** 

| User | Benefit |
|------|---------|
| **Tenants** | Check whether a listing is fairly priced |
| **Investors** | Target high-ROI features for purchasing decisions |
| **Developers** | Guide amenity investment with data-backed rankings |
| **Policymakers** | Understand affordability drivers across cities |

## ***Installation***

To set up the project locally:

1. Navigate to the project folder:
   ```bash
   cd CSE6242_Project
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   If needed, you can use:
   ```bash
   python -m pip install -r requirements.txt
   ```

## ***Execution***

Use the hosted prediction app here: [amenitiesmagnet.streamlit.app/predict](https://amenitiesmagnet.streamlit.app/predict)

Demo video with installation and execution walkthrough: [YouTube](https://www.youtube.com/watch?v=bY5u8SnmNXQ)

To launch the Streamlit dashboard:

1. Open a terminal in the project directory:
   ```bash
   cd CSE6242_Project
   ```
2. Run the application:
   ```bash
   streamlit run Code/app.py
   ```
3. The app will open in your browser. From there you can navigate to:
   - Map View
   - Explore Listings
   - Rental Price Estimator
   - Model Insights

---

## ***Members***   
*T. Buttrick · A. Diallo · D. Siachras · N. Gala · S. Venkatesh*

## ***Directory Structure:*** 
### Model Building
1. [Data](https://github.com/suprajaven/AmenitiesMagnet/tree/main/CSE6242_Project/Data) - Datasets used for model development, evaluation, and dashboard inputs.
2. [Code](https://github.com/suprajaven/AmenitiesMagnet/tree/main/CSE6242_Project/Code) - Application source code, data processing pipelines, and model implementation.
3. [Analysis](https://github.com/suprajaven/AmenitiesMagnet/tree/main/CSE6242_Project/Analysis) - Exploratory analysis, modeling experiments, and supporting documentation.
4. [Visualizations](https://github.com/suprajaven/AmenitiesMagnet/tree/main/CSE6242_Project/Visualizations) - Charts, figures, and other visual assets created for the project.


### Reporting:
1. [Project Proposal](https://github.com/suprajaven/AmenitiesMagnet/tree/main/CSE6242_Project/Project%20Proposal) - Initial project concept, scope, and planned approach.
2. [Proposal Video](https://github.com/suprajaven/AmenitiesMagnet/tree/main/CSE6242_Project/Proposal%20Video) - Video presentation accompanying the original proposal.
3. [Progress Report](https://github.com/suprajaven/AmenitiesMagnet/tree/main/CSE6242_Project/Progress%20Report) - Mid-project update covering progress, challenges, and next steps.
4. [Final Report](https://github.com/suprajaven/AmenitiesMagnet/tree/main/CSE6242_Project/Final%20Report) - Comprehensive summary of the methodology, results, and conclusions.
5. [Final Video](https://github.com/suprajaven/AmenitiesMagnet/tree/main/CSE6242_Project/Final%20Video) - Final presentation highlighting the completed project and key findings.
