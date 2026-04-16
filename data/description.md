In-vivo dataset: data/invivo_data.csv
- experiment_id: source CSV filename for each row (e.g., sheepAround1.csv)
- features: columns matching ratio_*
- label: fSaO2

Simulation dataset: data/simulation_data.processed.pkl
- features: numbers separated by underscores
- labels: texts in general without any numbers


Metadata file: data/dataset_metadata.yaml
- contains invivo and simulation entries
- each entry stores features, labels, detector_distances (aligned to features), and wavelength (aligned to features)
