from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load saved files
with open("dbscan_model.pkl", "rb") as f:
    dbscan = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("cluster_to_cities.pkl", "rb") as f:
    cluster_to_cities = pickle.load(f)

def assign_dbscan_label(model, X_scaled_single):
    if not hasattr(model, "components_") or len(model.components_) == 0:
        return -1
    core_samples = model.components_
    core_indices = model.core_sample_indices_
    labels = model.labels_
    dists = np.linalg.norm(core_samples - X_scaled_single, axis=1)
    nearest_idx = np.argmin(dists)
    nearest_dist = dists[nearest_idx]
    if nearest_dist <= model.eps:
        return int(labels[core_indices[nearest_idx]])
    else:
        return -1

@app.route("/", methods=["GET", "POST"])
def index():
    cluster_label = None
    places = []
    message = None

    if request.method == "POST":
        try:
            rating = float(request.form["rating"])
            best_time = request.form["best_time"]

            # Encode features
            best_time_encoded = encoder.transform([[best_time]])
            X = np.hstack(([[rating]], best_time_encoded))

            # Scale
            X_scaled = scaler.transform(X)

            # Predict cluster
            cluster_label = assign_dbscan_label(dbscan, X_scaled[0])

            # Get matching cities
            places = cluster_to_cities.get(cluster_label, [])

            if cluster_label == -1:
                message = "No matching cluster found (noise point)."
            else:
                message = f"Cluster {cluster_label} contains {len(places)} places."

        except Exception as e:
            message = f"Error: {e}"

    best_time_options = encoder.categories_[0].tolist()
    return render_template("index.html", message=message, places=places, best_time_options=best_time_options)

if __name__ == "__main__":
    app.run(debug=True)
