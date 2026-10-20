# Infrastructure

This directory is reserved for deployment configuration. A production deployment should schedule `trm_forecast.pipeline.daily_run`, store raw data in a versioned object store, keep credentials outside the repository, and publish API/dashboard services separately from model training.
