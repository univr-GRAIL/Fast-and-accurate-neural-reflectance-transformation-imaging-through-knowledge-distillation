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
   set the following variables:   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; data_path   = "your_dataset_path/"  &nbsp; # Path to your training dataset  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; output_path = "output/"          &nbsp;    # Directory for saving outputs  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; mask = True or False  &nbsp; # Enable/disable masking  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; src_img_type  # Image extension type, i.e., jpg [default], png, or tiff.

**To Test training and testing** download a dataset from <a href="https://univr-my.sharepoint.com/:f:/g/personal/tinsaegebrechristos_dulecha_univr_it/EkVPviXq86VGjixc6Ti18SoBdkKTOeaWqBlQzV09rpdHfg?e=cY54V6" text-decoration="none" target="_blank">**here** </a> and place inside modules folder.


### 2.  Run Training  
   Execute:  
   ```bash
python train.py --data_path data_path --ld_file ld_file --src_img_type src_img_type --output_path output_path
   ```  
After training, the outputs will be saved inside the output-path directory, within the Teacher and Student subdirectories. Each subdirectory contains:
- The trained model (.pth) 
- Compressed coefficients
- Image planes and a JSON file for OpenLIME visualization

## Test/Relighting  
### 1. Configure test parameters   
Modify the following variables:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model_path # Path to the trained model directory  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; light_path   # Path to the test light direction and mask path, if needed  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;masked # Use mask during relighting or not, default=False

### 2. Run:  
```bash
python test.py  --model_path model_path  --light_path light_path
```  

For example, if you set: 
```bash 
model_path=test_dataset/outputs/Student 
light_path = test_dataset/test_fig6 and
```
This will generate images shown in figure 6 on the paper and saves it in the relighted folder. 

but if you set:
```bash 
light_path = 'test_dataset/test'
``` 
Generates images relighted from 20 light directions defined in test_dataset/test/dirs.lp



## Datasets:

All the images(training, test, and relighted using different algorithms) can be found at 
<a href="https://univr-my.sharepoint.com/:f:/g/personal/tinsaegebrechristos_dulecha_univr_it/Er4M2DWps1FDjLce2Ssd3pYByROXPOKOeeYATFjhl261cQ?e=0ClcJ9" text-decoration="none" target="_blank"> DiskNeuralRTI Datasets </a>.

##  Evaluation / Metrics
For example, To reproduce Table 6 from the paper (Average LPIPS / ΔE for RealRTI relighting):

### 1. modify calculate_metrics.py
 Download RealRTI dataset from 
 <a href="https://univr-my.sharepoint.com/:f:/g/personal/tinsaegebrechristos_dulecha_univr_it/EjRfAl2DeppAsDLDo5rkr0gBg1-54GrN3WYzLIKQRu2yPg?e=fbv2tp" target="_blank"> here </a>.  
 
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
├── requirements.txt  
├── Calculate_metrics.py test.py 
├── test.py    train.py   
├── train.py  
├── README.md             # Project dependencies
├── modules/                 
    ├── dataset/                   
    ├── model/
    ├── relight/                  
    ├── utils/
    │   └── params.py             # Tra         
                   # Project documentation
```

## Citation

If you consider our work useful for your research please consider citing:

```bash
Coming soon
```
## License

This project is licensed under the MIT License with the Commons Clause.

You are free to use, modify, and distribute this code for non-commercial purposes. Commercial use is prohibited without obtaining a commercial license. For commercial use inquiries, please contact the authors.




