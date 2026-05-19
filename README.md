Pelkum Donahue & Inceoglu (2026): Calculating 3D Interplanetary Shock Velocity and Normals via Triple-Spacecraft XGBoost Optimization


**helper.py:** General methods used in project:
- LP_filter_data for lowpass signal filtering (used for automated shock selection)
- interpolate: used to unify satellite data to common cadence

**dsc_urls.py:** List of all wget urls for DSCOVR data

**1a_load_aw_data:** Retrieves and stores ACE and Wind data from NASA CDAWeb

**1b_load_dsc_data.py:** Retreives and stores DSCOVR data from NOAA portal

**1c_load_cme_sir_data.py:** Retrieves and stores CME and SIR event lists from CCMC and UCLA, repsectively

**2_prep_data:** Unifies, sychronizes, and saves ACE, Wind, and DSCOVR data cadences

**3a_generate_training_dists.py:** Fits Gaussian distributions to shock velocity components from the Helsinki shock database for model training


