# Weather-aware forecast fix plan

The dashboard currently calls `get_weather(city)` when generating a decision, but the RandomForest model is trained on synthetic data and the weather feature is not learned from the actual Kaggle sales dataset. This is why changing weather does not produce a trustworthy weather-driven forecast.

Planned fix:
- Train on actual `train.csv` data.
- Engineer calendar/weather features from the training pipeline.
- Use the selected city's weather forecast as a model input.
- Generate a real multi-month forecast series rather than repeating/scaling a single point estimate.
- Display historical and forecast demand with a clean line chart and uncertainty band.
- Keep SerpAPI market price as a separate external pricing signal.

This marker file is temporary and can be deleted after the full model integration is completed.