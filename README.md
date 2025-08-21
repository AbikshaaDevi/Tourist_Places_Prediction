# Tourist Places Prediction

## Overview
This repository implements a geographic clustering-based system to predict tourist places. It contains:

- `app.py` — application script for inference or interface.
- `Geo_Cluster_DBSCAN.ipynb` — notebook exploring and building the clustering model using DBSCAN.
- Model & utility files:
  - `dbscan_model.pkl` — trained clustering model.
  - `scaler.pkl` — preprocessing scaler.
  - `encoder.pkl` — categorical encoder.
  - `cluster_to_cities.pkl` — mapping of cluster labels to city or place identifiers.
- `holidify.csv` — dataset of tourist places or relevant features.
- `requirements.txt` — required Python dependencies.
- `templates/` — directory for UI templates or data presentation.

## File Structure
```
.
├── app.py
├── Geo_Cluster_DBSCAN.ipynb
├── holidify.csv
├── dbscan_model.pkl
├── scaler.pkl
├── encoder.pkl
├── cluster_to_cities.pkl
├── requirements.txt
└── templates/
```

## Getting Started

### 1. Install Dependencies
Run the following to install necessary packages:
```bash
pip install -r requirements.txt
```

### 2. Explore & Train
Open the notebook:
```bash
jupyter notebook Geo_Cluster_DBSCAN.ipynb
```
Use it to explore, preprocess data, run DBSCAN clustering, and generate the serialized model files.

### 3. Run the Application
After training (or with existing pickled models), launch the application:
```bash
python app.py
```
Check `app.py` for input requirements or API endpoints for getting tourist place predictions.

## Optional: Retrain Models
To retrain:
- Rerun the notebook to update clustering or encode logic.
- Serialize updated artifacts:  
  ```python
  # Example
  model.to_pickle("dbscan_model.pkl")
  scaler.to_pickle("scaler.pkl")
  encoder.to_pickle("encoder.pkl")
  ```
- Update `cluster_to_cities.pkl` as needed.

## Contributing
Contributions are welcome! Feel free to open issues or submit PRs for enhancements—e.g., better UI, improved clustering, additional data.

## License
No license is specified. To promote usage and contributions, consider adding a `LICENSE` file (e.g., MIT, Apache‑2.0).
