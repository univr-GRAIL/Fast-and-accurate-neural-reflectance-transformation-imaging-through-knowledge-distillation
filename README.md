# Fast-and-accurate-neural-reflectance-transformation-imaging-through-knowledge-distillation Implementation
This is a PyTorch implementation of Fast-and-accurate-neural-reflectance-transformation-imaging-through-knowledge-distillation. 

## Getting Started
Download  
```bash
git clone https://github.com/univr-GRAIL/Fast-and-accurate-neural-reflectance-transformation-imaging-through-knowledge-distillation.git

```
Change working directory
```bash
cd Fast-and-accurate-neural-reflectance-transformation-imaging-through-knowledge-distillation
```
## Create a virtual environment:   
```bash
python -m venv neuralenv  
```
### Activate a virtual environment:  
On Windows: 
```bash
   neuralenv\Scripts\activate  
```
On Linux/macOS:  
```bash
   source neuralenv/bin/activate  
```
## Install Dependencies:   
Make sure you're in a Python environment, neuralenv
Install the dependencies using the following command:    
```bash    
pip install -r requirements.txt  
```
## Training
###  1. Configure the training parameters
   Edit the file:  
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; utils/params.py
Set the paths and options:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; data_path   = "your_dataset_path/"  &nbsp; # Path to your training dataset  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; output_path = "output/"          &nbsp;    # Directory for saving outputs  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; mask = True or False  &nbsp; # Enable/disable masking

**To Test training and testing** download a dataset from <a href="https://univr-my.sharepoint.com/:f:/g/personal/tinsaegebrechristos_dulecha_univr_it/EkVPviXq86VGjixc6Ti18SoBdkKTOeaWqBlQzV09rpdHfg?e=cY54V6" target="_blank">**here** </a> and place inside modules folder.


### 2.  Run Training  
   Execute:  
   ```bash
     python train.py
   ```  
After training, the output/ directory will contain:
- The trained model (.pth) 
- Compressed coefficients (teacher & student)
- Image planes and a JSON file for OpenLIME visualization

If the codes run correctly, you will find the output file in the [output-path] folder  

## Test/Relighting  
### 1. **Edit** test.py   
Modify the following variables:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model_path = "output_path/model_name.pth" # Trained model path  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; test_data = "test_data_path"  # Folder containing test light direction, mask (if necessary), and ground truth image

### 2. Run:  
```bash
 python test.py  
```  

For example, if you set: 
```bash 
test_data = 'test_dataset/test_fig6'
```
This will generate images shown in figure 6 on the paper and saves it in the relighted folder. 

but if you set:
```bash 
test_data = 'test_dataset/test'
``` 
Generates images relighted from 20 light directions defined in test_dataset/test/dirs.lp



## Datasets:

All the images(training, test, and relighted using different algorithms) can be found at 
[DiskNeuralRTI Datasets](https://univr-my.sharepoint.com/my?id=%2Fpersonal%2Ftinsaegebrechristos%5Fdulecha%5Funivr%5Fit%2FDocuments%2Fdatasets&viewid=23a4d494%2D5c4d%2D4800%2D8a0f%2D56d3734699cd)


##  Evaluation / Metrics
For example, To reproduce Table 6 from the paper (Average LPIPS / ΔE for RealRTI relighting):

### 1. modify calculate_metrics.py
 Download RealRTI dataset from [here](https://univr-my.sharepoint.com/my?id=%2Fpersonal%2Ftinsaegebrechristos%5Fdulecha%5Funivr%5Fit%2FDocuments%2Fdatasets&viewid=23a4d494%2D5c4d%2D4800%2D8a0f%2D56d3734699cd)  
 set parent_folder = ['RealRTI dataset folder']
 
### 2. Run
``` bash
cd modules\utils
python calculate_metrics.py
```
## 📁 Project Structure

```bash
DiskNeuralRTI/
│
├── requirements.txt              # Project dependencies
│
└── modules/                 
    ├── dataset/                   
    ├── model/
    ├── relight/                  
    ├── test_dataset/  
    ├── utils/
    │   └── params.py             # Training parameters and paths
    ├── Calculate_metrics.py test.py 
    ├── test.py    train.py   
    ├── train.py               
    └── README.md                 # Project documentation





