"""Simple CNN for filling cloud gaps in GlobColour L3 chlorophyll (Indian Ocean, IO.zarr).

Modules
  data      subset IO.zarr -> cached npz; train/val/test split; normalisation
  clouds    fake-cloud masks for training and evaluation
  model     the 3-layer CNN (port of the geo-smart Simple CNN baseline)
  train     training loop + CLI  (python -m chl_cnn.train --help)
  evaluate  metrics on hidden pixels, reference predictors
  plotting  colourblind-friendly figures used by the notebooks
"""
