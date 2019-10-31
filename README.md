# DeepPurity
A tool to estimate tumor purity of tumor samples based on the deep learning model  
***Note**: This project has stopped due to a lack of cancer genome data for the deep learning.*

The documentation below is the previous documentation of this project.
***

## **Dependencies**  
python >= 3.6.5  
tensorflow (https://www.tensorflow.org/install/)  
keras (https://keras.io/#installation)  
pysam (https://pysam.readthedocs.io/en/latest/installation.html)  
numpy (https://scipy.org/install.html)  
pandas (https://pandas.pydata.org/pandas-docs/stable/install.html)  
pickle (https://docs.python.org/3/library/pickle.html)  

## Installation  
For install python >= 3.6.5, visit (https://www.anaconda.com/download/#linux) and download python >= 3.6.5  
Install anaconda and above modules for executing DeepPurity.  

    $ bash Anaconda3-5.3.1-Linux-x86_64.sh
    $ ${Anaconda_PATH}/bin/pip install tensorflow keras pysam numpy pandas pickle
    
    
## **Usage**  
We pre-trained a convolutional neural network model with the trainig data and provided the best model for predicting the tumor purity level. DeepPurity should be executed at the same directory where DeepPurity is installed. Then DeepPurity results are at the same directory. [Image_list_file] is '{DeepPurity_PATH}/ref/[Sample_name].txt'.  
 
Input: tumor bam, normal bam, mutect result from tumor and normal bam (mto format)  
Command:  
    
     ## Command for making images for predicting tumor purity level  
     ## This command makes 1,000 position files at '{DeepPurity_PATH}/[Sample_name]/pos/[Sample_name]_positions_*.tsv' and 1,000 image objects at '{DeepPurity_PATH}/[Sample_name]/images/[Sample_name]_image_*.pkl'. DeepPurity makes image list file at '{DeepPurity_PATH}/ref/[Sample_name].txt'.  
     $ ./DeepPurity MakeImage –t [Tumor_bam] –n [Normal_bam] –m [Mutect_result] –s [Sample_name]  
     
     ## Optional, Train deep learning model with user data  
     ## This command makes user training model at '{DeepPurity_PATH}/model/[Sample_name].hdf5'.  
     $ ./DeepPurity Train -s [Sample_name]  
     
     ## Predict the tumor purity level  
     ## If '-b' option is not entered, DeepPurity uses our default model for predicting the tumor purity level. This command makes the prediction results at '{DeepPurity_PATH}/prediction/[Sample_name].txt'.  
     $ ./DeepPurity Predict -b [User_model_sample_name (optional)] -s [Sample_name]  

## Quick start  

Input: './HCC1954.TUMOR.compare.bam' (tumor bam), './HCC1954.NORMAL.compare.bam' (normal bam), './hcc1954.mto' (mutect result)  
This example makes ./prediction/HCC1954-test.txt result.  

    $ ./DeepPurity MakeImage -t ./HCC1954.TUMOR.compare.bam -n ./HCC1954.compare.bam -m ./hcc1954.mto -s HCC1954-test
    $ ./DeepPurity Predict -s HCC1954-test

## User trainig mode  

Input: './HCC1954.TUMOR.compare.bam' (tumor bam), './HCC1954.NORMAL.compare.bam' (normal bam), './hcc1954.mto' (mutect result)  
This example makes ./prediction/HCC1954-test.txt result.  

    $ ./DeepPurity MakeImage -t ./HCC1954.TUMOR.compare.bam -n ./HCC1954.compare.bam -m ./hcc1954.mto -s HCC1954-test
    ## (Optional)
    $ ./DeepPurity Train -s HCC1954-test
    ## (Optional)
    $ ./DeepPurity Predict -b ./model/HCC1954-test.hdf5 -s HCC1954-test

