.PHONY: all install etl features models cv geo analysis

all: etl features models cv geo analysis

install:
	pip install -r requirements.txt

etl:
	python src/etl/merge_dvf.py
	python src/etl/filter_antibes.py
	python src/etl/aggregate_monthly.py
	python src/etl/merge_dvf_geo.py

features:
	python src/features/build_features.py
	python src/features/build_features_geo.py

models:
	python src/models/lstm.py
	python src/models/gru.py
	python src/models/transformer.py

cv:
	python src/models/cv_lstm.py
	python src/models/cv_compare.py
	python src/models/cv_ensemble.py

geo:
	python src/models/lstm_geo.py
	python src/models/forecast_geo.py
	python src/models/transformer_geo.py
	python src/models/forecast_transformer_geo.py
	python src/scoring/score.py

analysis:
	python src/analysis/eda.py
	python src/analysis/plot_results.py
	python src/analysis/heatmap.py
	python src/analysis/heatmap_forecast.py
	python src/analysis/heatmap_forecast_compare.py
