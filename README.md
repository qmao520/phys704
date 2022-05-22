# phys704

Classification of Quasars and Non-Quasars Object using machine learning
=======

- cleaning_notebooks folder:
    - data_cleaning_redQ: clean “W2M-red_qasars.csv” and output “data/red_qso.csv”
    - data_cleaning_v2_fainter: output the final set for the model, output dataset “data/train_qso_wo_petro_vs.csv”
    - faint_galaxy: cleaned the faint galaxy dataset, output dataset “data/final_galaxy_set2.csv” 
    
- data
    - raw: include all the original dataset I downloaded directly from canvas (fits and csv) or cleaned in manually in excel
    - combined: used TOPCAT to cross-match
    - output_from_CleanNotebook_folder
    
- ML_notebook
    - code for constructing the model & training the model
- saved_model
    - I picked (aka saved the trained model) from ML_notebook and run it here again for the results. 
    - file rfc_RScv_faint.sav is the saved final model that used for prediction
- Prediction
    - data_clean_prediction.ipynb: output a dataset that is ready to input into the trained model —> “ready_pred.csv”
    - prediction.ipynb: output the final dataset into result folder
    
    

To access the trained model:
    1) go to folder "saved_models" --> folder "faint"
    2) go to notebook "Models_faint.ipynb"
    3
