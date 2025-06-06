# Transit-Time Modeling Using Precipitation and Soil Moisture-Driven SAS Functions  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  

## Introduction  
This repository provides Python code for a **catchment-scale tracer transport model**. The model simulates hydrologic transport and tracer dynamics using **StorAge Selection (SAS) functions** to estimate **time-variable transit time distributions (TTDs)**. It integrates SAS formulations into a process-based hydrological model, enabling the joint simulation of water fluxes and isotopic tracer signals.

### SAS Function Formulations  
Two SAS model variants are implemented:

1. **`Tracer_Mod_Wettness.py`**  
   - Time-variable SAS shape for the **root zone** depends on the ratio of current storage to maximum storage, representing a soil-moisture-dependent preference for younger water.

2. **`Tracer_Mod_Ptresh_Wetness.py`**  
   - Extends the above by **adding a rainfall intensity threshold (`Ptresh`)**:  
     - If precipitation intensity exceeds `Ptresh`, young water is preferentially released, **regardless of wetness state**.  
     - If below the threshold, the soil-moisture-based formulation from `Tracer_Mod_Wettness.py` is used.

These formulations allow the model to represent **preferential flow activation** dynamically, controlled by both **soil moisture** and **precipitation intensity**.

---

## Reference  
This model supports the findings of the following publication:

> Türk, H., Stumpp, C., Hrachowitz, M., Schulz, K., Strauss, P., Blöschl, G., & Stockinger, M. (2024). *Soil moisture and precipitation intensity jointly control the transit time distribution of quick flow in a flashy headwater catchment*. Hydrology and Earth System Sciences Discussions, 1–33. https://doi.org/10.5194/hess-2024-359

---

## Repository Structure  
- `Data/`: Contains test data (`Data_test.csv`) for sample model runs  
- `Model_Run.py`: Example script for model calibration and tracer simulation based on two model structure
- `Tracer_Mod_Wettness.py`: Model variant where root zone SAS depends on soil moisture  
- `Track_Tracer_Mod_Wettness.py`: Model variant allowing manual adjustment of SAS parameters for root zone and groundwater.  Useful for testing sensitivity and scenario analysis.
- `Tracer_Mod_Ptresh_Wetness.py`: Model variant where the root zone SAS function depends on soil moisture, additionally prioritizes young water release when precipitation intensity exceeds a defined threshold (`Ptresh`).
- `Track_Tracer_Mod_Ptresh_Wetness.py`: Model variant allowing manual adjustment of SAS shape parameters for both the root zone and groundwater compartments. Useful for testing sensitivity and scenario analysis.

---

## Running the Model  
To run a basic test with sample data, execute:

```bash
python Model_Run.py

