.PHONY: preprocess train-long-term train-pems train-imputation figures evaluate test

preprocess:
	bash scripts/preprocess_all.sh

train-long-term:
	bash scripts/train_long_term.sh

train-pems:
	bash scripts/train_pems.sh

train-imputation:
	bash scripts/train_imputation.sh

figures:
	bash scripts/reproduce_manuscript_figures.sh

evaluate:
	bash scripts/evaluate.sh

test:
	python -m pytest tests
