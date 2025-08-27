import argparse
import os
import numpy as np
import cv2
from glob import glob
import torch
import torch.nn.functional as F
from models.BRMSANet import BRMSANet
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, img_paths, mask_paths, test_size):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.test_size = test_size
                
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
                
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)  

        name = self.img_paths[idx].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
                        
        image = cv2.resize(image, (self.test_size, self.test_size))          
            
        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:, :, np.newaxis]
        mask = mask.astype('float32') / 255
        mask = mask.transpose((2, 0, 1))        
        
        return name, np.asarray(image), np.asarray(mask)   
    
def inference(model, test_loader, result_path):  
    
    model.eval()      
    
    for i, pack in enumerate(test_loader, start=1):
        name, image, gt = pack
        gt = gt[0][0]
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda() 

        res, _, _, _ = model(image)                
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)   
           
        Image.fromarray(((res > .5) * 255).astype(np.uint8)).save(os.path.join(result_path, name[0]))       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str,
                        default='b3')
    parser.add_argument('--weight', type=str,
                        default='')    
    parser.add_argument('--test_size', type=int,
                        default=352, help='training dataset size')    
    parser.add_argument('--task_folder', type=str,
                        default='polyp', help='path to best checkpoint')
    parser.add_argument('--test_path', type=str,
                        default='', help='path to dataset')
    args = parser.parse_args()   

    save_path = './results/{}/'.format(args.task_folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        print("Save path existed")       
    
    model = BRMSANet().cuda() 
    ckpt_file = './snapshots/{}/{}'.format(args.task_folder, 'best.pth')
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint['state_dict'])        
    
    for data_name in ['CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        data_path = './data/{}/{}'.format(args.test_path, data_name)
        test_img_paths = glob('{}/images/*'.format(data_path))
        test_mask_paths = glob('{}/masks/*'.format(data_path))
        test_img_paths.sort()
        test_mask_paths.sort()
        test_dataset = Dataset(test_img_paths, test_mask_paths, args.test_size)
        test_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=1,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False
                    )  
        
        result_path = save_path + data_name
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        inference(model, test_loader, result_path)
        
    
    

