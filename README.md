# **SUPERVISED BINARY IMAGE CLASSIFICATION USING CNNs - PYTORCH**

This repository contains the code written during my internship in data analysis and machine learning.    

The data consists of simulated particle track (event) projections from three readout planes.    

Directory structure:  
<blockquote>   

     PATH/ 
        |-- PLANE_0/ 
            |-- CLASS_0/ 
            |-- CLASS_1/  
        |-- PLANE_1/  
            |-- CLASS_0/  
            |-- CLASS_1/
        |-- PLANE_2/
   	    	|-- CLASS_0/
  	    	|-- CLASS_1/
</blockquote>   
  

Workflow: 

`norm_to_array_ALL.py`: All files were extracted and normalized to facilitate processing by CNNs. My project began with the conversion of normalized data, saving the sparse matrices as .npz files.    

`ROI_values_count_ALL.ipynb` and `npz_cleaning.ipynb`: Empty or very small files were removed to ensure data quality.  

`txt_filenames_for_dataset.ipynb`: Text files were generated, containing filenames and the associated label for each event. Each row in the text file contains three filenames corresponding to the event projections from the three readout planes. The filenames were randomized and split into two files: 70% for the training dataset and 30% for the testing dataset. The use of two datasets containing identical images enables a fair comparison between models and approaches.  

`EARLY_FUSION SCRIPT.py` and `LATE_FUSION_SCRIPT.py`: Two distinct approaches were implemented:  
-	Early Fusion: All event data was combined into a single tensor, and thus one model was trained using all three readout planes simultaneously.
-	Late Fusion: Each model was trained using data from a single readout plane. After training, the models were combined to classify the events.  

For both approaches, models were trained using the following architectures: AlexNet, VGG11, and VGG19.  

Metrics such as loss and accuracy were recorded for both the training and the validation phases. The models were saved after each epoch.  

`plotting_loss_accuracy.ipynb`:  Post-training, the loss and accuracy metrics were visualized for each approach and model. Based on the best loss/accuracy ratio, a subset of models was selected for subsequent testing and evaluation.  

`testing.ipynb`: During the testing phase, key metrics required for plotting the ROC curve were saved in .pkl format.  

`roc_curve_ef&lf.ipynb`: In the final step, the data saved during testing was used to generate the confusion matrix, classification report, and the ROC curve for each approach and each model.
