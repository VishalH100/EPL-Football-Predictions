# EPL-Football-Predictions
1. Read *5 seasons' data* from datasets directory (Source: Football-data.com) and merge into 1 combined file, then merge with *EPL 2018-2023* data (Source: Kaggle.com) and store it into preprocessed directory.
   - open and run ```Data Preprocessing.ipynb``` in Jupyter Notebook
2. Generate Predictions:
   - `python Predictions_using_GB.py`
   - `python Predictions_using_RF.py`
   - `python Predictions_using_RNN.py`
3. Generate Final Standings
   - open and run `Data Viz.ipynb` in Jupyter Notebook
4. Launch WebApp using streamlit in terminal
   - `streamlit run UI.py`
 
