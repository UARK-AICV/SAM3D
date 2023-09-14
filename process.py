import pickle
import nibabel as nib
import numpy as np  
import glob


for i in glob.glob("/home/hdhieu/3DSAM-Decoder-1/DATASET_Acdc/sam3d_raw/sam3d_raw_data/Task01_ACDC/imagesTs/*"):
    img = nib.load(i)
    img = img.get_fdata()
    x = np.where(img == 2)
    # print("image: ", i)
    print("image shape: ", img.shape)
    # print("gal: ", x)
# img = nib.load("/home/hdhieu/3DSAM-Decoder-1/DATASET_Synapse/sam3d_raw/sam3d_raw_data/Task002_Synapse/labelsTs/label0035.nii.gz")

# img = img.get_fdata()

# array = np.array(img)

# print(img)

# x = np.where(img == 4)

# print(x)



# with open('/home/bntan/3DSAM-Decoder-1/DATASET_Acdc/sam3d_raw/sam3d_raw_data/Task01_ACDC/Task001_ACDC/sam3d_Plansv2.1_plans_3D.pkl', 'rb') as f:
#     data = pickle.load(f)
    

# print(data['normalization_schemes'])