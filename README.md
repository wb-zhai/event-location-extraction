# event-location-extraction
A test repo for event-location-extraction with GLiNER2.

Run with:
1. `uv run extract.py`, this is the most basic version without risk factors
2. `uv run extract_with_risk_factors.py` included risk factors defined after https://www.science.org/doi/10.1126/sciadv.abm3449. The problem here is that it only extracts one single event. 
3. `extract_with_risk_factors_v2.py` is an attempt to extract more event-location entities without success.

All files output the results as `gliner2_extraction_results.json`.
