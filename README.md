## Pelkum Donahue & Inceoglu (2026): Calculating 3D Interplanetary Shock Velocity and Normals via Triple-Spacecraft XGBoost Optimization

Authors: K. Pelkum Donahue (writing), F. Inceoglu (editing and review)

----

**Python files displayed in boldface**

_Data files displayed in italics_

----

**helper:** General methods used in project:
- LP_filter_data for lowpass signal filtering (used for automated shock selection)
- interpolate: used to unify satellite data to common cadence

**dsc_urls:** List of all wget urls for DSCOVR data

**1a_load_aw_data:** Retrieves and stores ACE and Wind data from NASA CDAWeb (references in main text)

**1b_load_dsc_data:** Retreives and stores DSCOVR data from NOAA portal (reference in main text)

**1c_load_cme_sir_data:** Retrieves and stores CME and SIR event lists from CCMC and UCLA, repsectively (references in main text)

**2_prep_data:** Unifies, sychronizes, and saves ACE, Wind, and DSCOVR data cadences

**3a_generate_training_dists:** Fits Gaussian distributions to shock velocity components from the Helsinki shock database for model training (reference in main text)

**3b_generate_training_data:** Generates orbital artifact correction training data by sending synthetic plane waves from Helsinki distributions through real 3-sat orbital geometries

**4_train_xgb:** Trains XGBoost model on synthetic earthward and sunward shocks, separately, evaluating on test set, and saving models
- _earthward_xgb_model.pkl_ and _sunward_xgb_model.pkl_: saved XGBoost models

**4a_xgb_viz:** Visualizes test set performance via hemisphere-separated heatmaps, speed histogram, and plane projections of velocity vectors

**5a_step_finder:** Iterates over all 7 years of data, applying automated logistic regression to identify shock fronts passing all three satellites, and saving list

**5b_prop_calc:** Performs plane wave least-squares fit for all identified shock windows to calculate shock velocity vectors

**6_transform_data:** Sends calculated velocity vectors with orbital parameters through XGBoost model from 4_train_xgb to remove orbital artifact
- _corrected_l1_cases.pkl_ is list of all L1 shocks, containing both the raw and XGBoost-corrected velocity vectors
  - 'start': shock start time (datetime)
  - 'tot_window': duration of shock analysis window (min)
  - 'direction': FF vs FR (string)
  - 'sat_*': order of satellites to detect shock (string)
  - 't_a': number of minutes after window start of first detection (int)
  - 'lag_ab': time lag between 1st and 2nd detections (int)
  - 'lag_ac': time lag between 1st and 3rd detections (int)
  - 'pos_*': cartesian GSE positions of each satellite (Re)
  - 'orb_norm_*': cartesian GSE components of normal vector to plane containing all three satellites (unitless)
  - 'v_calc_*': cartesian GSE components of least-squares shock velocity vector (km/s)
  - 'v_corrected': cartesian GSE vector of XGBoost-corrected shock velocity vector (km/s)

**7_visualize_transform:** Visualizes xgboost model effect on 3-sat L1 shocks through hemisphere-separated heatmaps, speed histograms, 2d plane projections, and a 3D normal vector plot

**8a_triple_coherence_data:** Iterates through all shocks and calculates signal coherence of B-field measurements, time-lagged according to the results of the automated logistic regression

**8b_visualize_coherence:** Bins shock observations by distance from first satellite (equal number of shocks per bin) for general coherence, plotting falloff

**8c_coherence_lengths:** Fits exponential decay functions to falloffs from prior script to calculate characteristic coherence decay length

**9_match_drivers:** Iterates through L1 shock cases, searching for arrivals of CMEs and SIRs and saving any found associations

**10_val_normals:** Using shock normals from this study's L1 list, Helsinki list, and Harvard list, send plane waves back through the correct orientations to check observed vs predicted arrival times, plotting results with CCC 1:1 linearity scores
