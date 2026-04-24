# Order of running:
1. `data_exploration.ipynb`: Does a basic exploration of dataset (and loads it)
2. `data_cleaning.ipynb`: Does a basic cleaning of useless *NA* and a bit of feature engineering (saves the cleaning set)
3. `modelisation.ipynb`: Shows clustering prowess on basic dataset (compared to `GRAVITE_3`) and tests basic models for classification



# TODO: 
## `modelisation.ipynb`
- ~~create `clustering.py` that runs and shows multiple clustering methods to the notebook (w/ a 3d scatter plot)~~ \
MDS and Isomap need distances (but we have 200k accidents, so that is too compute heavy)
- ~~write `preprocessing` that **removes unwanted features** found in exploration/cleaning and then preprocesses those~~
- ~~create end pipeline that compare the ^ removed feature dataset w/ vs w\ preprocessing and/or dimension reduction (using the chosen method before)~~
- ~~update: create pipeline that checks for performance `for k in PCA(k)`, `for strategy in strategies`)~~
- analyse it all <3
