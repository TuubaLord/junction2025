# junction2025




start with: BRRD, EBA, FIVA_MOK/EN

# PROCESS
Parse -> filter -> split -> bins -> profit

## 1. Parse

parse_EBA.py and parse_fiva_mok.py need to be run inside the corresponding folders (EBA, FIVA_MOK). They create the json in wanted format

## 2. Filter unrelated articles

select_relevant.py takes an input file and returns 2 output files (relevant and unrelevant). The file is configured in the code

## 3. Split to risk category

split_by_risk_category.py takes an input file (preferable relevant articles given by the filter before) and writes out 5 different files, 1 for each risk. the file names are configured in the code