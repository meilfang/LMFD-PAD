import os
import torch
from model.mfad import FAD_HAM_Net

import cv2
import albumentations
from albumentations.pytorch import ToTensorV2

PRE__MEAN = [0.485, 0.456, 0.406]
PRE__STD = [0.229, 0.224, 0.225]

# normalize image and crop as size 224xx224
def TestDataAugmentation():
    transform_val = albumentations.Compose([
        albumentations.SmallestMaxSize(max_size=256),
        albumentations.CenterCrop(height=224, width=224),
        albumentations.Normalize(PRE__MEAN, PRE__STD),
        ToTensorV2(),
        ])
    return transform_val

def test(image_path, model_path, threshold):
    # read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = TestDataAugmentation()(image=image)['image']
    image = image.unsqueeze(dim=0)
    # load model
    model = FAD_HAM_Net(pretrain=False, variant='resnet50')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    pred, _ = model(image)
    pred = torch.sigmoid(pred)

    # compare with threshold
    if pred[0] >= threshold:
        label = 'bonafide'
    else:
        label = 'attack'

    # output
    print(f'{image_path} is detected as {label}. The prediction score is {pred[0].detach().numpy()}')

if __name__ == "__main__":
    # Each pretrained model has different threshold, please use the correct threshold of corresponding pretrained model
    icm_o_th, ocm_i_th, omi_c_th, oci_m_th = 0.7309441, 0.6971898, 0.613508, 0.53312653
    # The input image should be detected face image
    model_path = 'trained_weights/icm_o.pth'
    image_path = 'testing_images/2_3_36_2_1.png'
    test(image_path, model_path, threshold=icm_o_th)
    # for verification: the score for test image 2_3_36_2_1.png (gt label is attack) is 0.5000488, and for 2_3_36_1_1.png (bona fide) is 0.73105854.
