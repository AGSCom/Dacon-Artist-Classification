# Private 10위(0.83173) Efficientnet + Ensemble

- 먼저 Pytorch로 Efficientnet B1-B6 모델을 학습하여 앙상블 진행
- 다음 TensorFlow로 Efficientnet B1-B6 모델을 학습하여 앙상블 진행
- 최종 적으로 Pytorch 모델과 TensorFlow 모델을 앙상블 하여 결과 도출.


1. Pytorch Model
    - 1. import Library
    - 2. Make PyTorch Dataset
    - 3. Make PyTorch Model
    - 4. Train PyTorch Model
    - 5. Test PyTorch Model
2. Tensorflow Model
    - 1. import Library
    - 2. Make TF Dataset
    - 3. Make TF Model
    - 4. Train TF Model
    - 5. Test TF Model
3. Ensemble PyTorch & TensorFlow
4. Make Submission

## Pytorch Model

### 1.Import Library


```python
import random
import pandas as pd
import numpy as np
import os
import cv2

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torchvision.models as models

from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore') 
import torch, gc

from torch.optim  import lr_scheduler
```


```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```


```python
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':30,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32,
    'SEED':41
}
```


```python
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정
```

### 2. Make PyTorch Dataset


```python
os.getcwd()
```




    'D:\\ML\\Artist Classification'




```python
base_dir = 'D:/ML/Artist Classification'
data_dir = 'D:/ML/Artist Classification/open'


df = pd.read_csv(f'{data_dir}/train.csv')
df1 = pd.read_csv(f'{data_dir}/train.csv')

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>img_path</th>
      <th>artist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>./train/0000.jpg</td>
      <td>Diego Velazquez</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>./train/0001.jpg</td>
      <td>Vincent van Gogh</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>./train/0002.jpg</td>
      <td>Claude Monet</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>./train/0003.jpg</td>
      <td>Edgar Degas</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>./train/0004.jpg</td>
      <td>Hieronymus Bosch</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Label Encoding
le = preprocessing.LabelEncoder()
df['artist'] = le.fit_transform(df['artist'].values)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>img_path</th>
      <th>artist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>./train/0000.jpg</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>./train/0001.jpg</td>
      <td>48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>./train/0002.jpg</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>./train/0003.jpg</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>./train/0004.jpg</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df, val_df, _, _ = train_test_split(df, df['artist'].values, test_size=0.2, random_state=CFG['SEED'])
```


```python
train_df = train_df.sort_values(by=['id'])
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>img_path</th>
      <th>artist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>./train/0000.jpg</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>./train/0002.jpg</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>./train/0003.jpg</td>
      <td>10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>./train/0005.jpg</td>
      <td>38</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>./train/0006.jpg</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>




```python
val_df = val_df.sort_values(by=['id'])
val_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>img_path</th>
      <th>artist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>./train/0001.jpg</td>
      <td>48</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>./train/0004.jpg</td>
      <td>24</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>./train/0017.jpg</td>
      <td>10</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>./train/0021.jpg</td>
      <td>29</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>./train/0029.jpg</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>




```python
def get_data(df, infer=False):
    if infer:
        return df['img_path'].values
    return df['img_path'].values, df['artist'].values
```


```python
train_img_paths, train_labels = get_data(train_df)
val_img_paths, val_labels = get_data(val_df)
```


```python
print(f"{data_dir}/{train_img_paths[0].split('/')[-2]}/{train_img_paths[0].split('/')[-1]}")
```

    D:/ML/Artist Classification/open/train/0000.jpg
    


```python
class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(f"{data_dir}/{img_path.split('/')[-2]}/{img_path.split('/')[-1]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image
    
    def __len__(self):
        return len(self.img_paths)
```


```python
def resize_transform(size, state='train'):
    if state == 'train':
        transform = A.Compose([
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                                A.Rotate(p=0.5),
                                A.RandomRotate90(p=0.5),
                                A.RandomResizedCrop(height=size, width=size, scale=(0.3, 1.0)),
                                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ToTensorV2(),
                                ])
    else:
        transform = A.Compose([
                            A.Resize(size,size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

    return transform
# 256, 288, 320, 384, 456, 528
# 240, 260, 300, 380, 456, 528
```

### 3.Make PyTorch Model


```python
class BaseModel(nn.Module): # 224
    def __init__(self, num_classes=len(le.classes_)):
        super(BaseModel, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)

        self.classifier = nn.Sequential(nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(1000, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class EfficientNet_B1(nn.Module): # 256
    def __init__(self, num_classes=len(le.classes_)):
        super(EfficientNet_B1, self).__init__()
        self.backbone = models.efficientnet_b1(pretrained=True)

        self.classifier = nn.Sequential(
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(1000, num_classes))
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class EfficientNet_B2(nn.Module): # 288
    def __init__(self, num_classes=len(le.classes_)):
        super(EfficientNet_B2, self).__init__()
        self.backbone = models.efficientnet_b2(pretrained=True)

        self.classifier = nn.Sequential(
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(1000, num_classes))
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class EfficientNet_B3(nn.Module): # 320
    def __init__(self, num_classes=len(le.classes_)):
        super(EfficientNet_B3, self).__init__()
        self.backbone = models.efficientnet_b3(pretrained=True)

        self.classifier = nn.Sequential(
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(1000, num_classes))
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
class EfficientNet_B4(nn.Module): # 384
    def __init__(self, num_classes=len(le.classes_)):
        super(EfficientNet_B4, self).__init__()
        self.backbone = models.efficientnet_b4(pretrained=True)

        self.classifier = nn.Sequential(
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(1000, num_classes))
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class EfficientNet_B5(nn.Module): # 456
    def __init__(self, num_classes=len(le.classes_)):
        super(EfficientNet_B5, self).__init__()
        self.backbone = models.efficientnet_b5(pretrained=True)

        self.classifier = nn.Sequential(
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(1000, num_classes))
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class EfficientNet_B6(nn.Module): # 528
    def __init__(self, num_classes=len(le.classes_)):
        super(EfficientNet_B6, self).__init__()
        self.backbone = models.efficientnet_b6(pretrained=True)

        self.classifier = nn.Sequential(
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(1000, num_classes))
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x   
```

### 4.Train PyTorch Model


```python
def train(model,epochs, optimizer, train_loader, test_loader, scheduler, device):
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    
    best_score = 0
    best_model = None
    
    for epoch in range(1,epochs+1):
        model.train()
        train_loss = []
        for img, label in tqdm(iter(train_loader)):
            img, label = img.float().to(device), label.type(torch.LongTensor).to(device)
            
            optimizer.zero_grad()

            model_pred = model(img)
            
            loss = criterion(model_pred, label)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        tr_loss = np.mean(train_loss)
            
        val_loss, val_score = validation(model, criterion, test_loader, device)
            
        print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val F1 Score : [{val_score:.5f}]')
        
        if scheduler is not None:
            scheduler.step()
            
        if best_score < val_score:
            best_model = model
            best_score = val_score
        
    return best_model
```


```python
def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")

def validation(model, criterion, test_loader, device):
    model.eval()
    
    model_preds = []
    true_labels = []
    
    val_loss = []
    
    with torch.no_grad():
        for img, label in tqdm(iter(test_loader)):
            img, label = img.float().to(device), label.type(torch.LongTensor).to(device)
            
            model_pred = model(img)
            
            loss = criterion(model_pred, label)
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
            
        
    val_f1 = competition_metric(true_labels, model_preds)

    return np.mean(val_loss), val_f1
```


```python
num_epoch  = 30
```


```python
train_dataset_0 = CustomDataset(train_img_paths, train_labels, resize_transform(224,'train'))
train_loader_0 = DataLoader(train_dataset_0, batch_size = 32, shuffle=True, num_workers=0)

val_dataset_0 = CustomDataset(val_img_paths, val_labels, resize_transform(224,'test'))
val_loader_0 = DataLoader(val_dataset_0, batch_size=32, shuffle=False, num_workers=0)

model_0 = BaseModel()
optimizer_0 = torch.optim.Adam(params = model_0.parameters(), lr = CFG["LEARNING_RATE"])
scheduler_0 = lr_scheduler.MultiStepLR(optimizer_0, milestones=[10,20], gamma=0.5)
infer_model_0 = train(model_0,num_epoch, optimizer_0, train_loader_0, val_loader_0, scheduler_0, device)
```

    Downloading: "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth" to C:\Users\DELL/.cache\torch\hub\checkpoints\efficientnet_b0_rwightman-3dd342df.pth
    


      0%|          | 0.00/20.5M [00:00<?, ?B/s]



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_30064\3377559122.py in <cell line: 7>()
          5 val_loader_0 = DataLoader(val_dataset_0, batch_size=32, shuffle=False, num_workers=0)
          6 
    ----> 7 model_0 = BaseModel()
          8 optimizer_0 = torch.optim.Adam(params = model_0.parameters(), lr = CFG["LEARNING_RATE"])
          9 scheduler_0 = lr_scheduler.MultiStepLR(optimizer_0, milestones=[10,20], gamma=0.5)
    

    ~\AppData\Local\Temp\ipykernel_30064\2142813306.py in __init__(self, num_classes)
          2     def __init__(self, num_classes=len(le.classes_)):
          3         super(BaseModel, self).__init__()
    ----> 4         self.backbone = models.efficientnet_b0(pretrained=True)
          5 
          6         self.classifier = nn.Sequential(nn.ReLU(),
    

    ~\anaconda3\envs\mytorch\lib\site-packages\torchvision\models\_utils.py in wrapper(*args, **kwargs)
        140             kwargs.update(keyword_only_kwargs)
        141 
    --> 142         return fn(*args, **kwargs)
        143 
        144     return wrapper
    

    ~\anaconda3\envs\mytorch\lib\site-packages\torchvision\models\_utils.py in inner_wrapper(*args, **kwargs)
        226                 kwargs[weights_param] = default_weights_arg
        227 
    --> 228             return builder(*args, **kwargs)
        229 
        230         return inner_wrapper
    

    ~\anaconda3\envs\mytorch\lib\site-packages\torchvision\models\efficientnet.py in efficientnet_b0(weights, progress, **kwargs)
        755 
        756     inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    --> 757     return _efficientnet(inverted_residual_setting, 0.2, last_channel, weights, progress, **kwargs)
        758 
        759 
    

    ~\anaconda3\envs\mytorch\lib\site-packages\torchvision\models\efficientnet.py in _efficientnet(inverted_residual_setting, dropout, last_channel, weights, progress, **kwargs)
        370 
        371     if weights is not None:
    --> 372         model.load_state_dict(weights.get_state_dict(progress=progress))
        373 
        374     return model
    

    ~\anaconda3\envs\mytorch\lib\site-packages\torchvision\models\_api.py in get_state_dict(self, progress)
         61 
         62     def get_state_dict(self, progress: bool) -> Dict[str, Any]:
    ---> 63         return load_state_dict_from_url(self.url, progress=progress)
         64 
         65     def __repr__(self) -> str:
    

    ~\anaconda3\envs\mytorch\lib\site-packages\torch\hub.py in load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)
        725             r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
        726             hash_prefix = r.group(1) if r else None
    --> 727         download_url_to_file(url, cached_file, hash_prefix, progress=progress)
        728 
        729     if _is_legacy_zip_format(cached_file):
    

    ~\anaconda3\envs\mytorch\lib\site-packages\torch\hub.py in download_url_to_file(url, dst, hash_prefix, progress)
        613                   unit='B', unit_scale=True, unit_divisor=1024) as pbar:
        614             while True:
    --> 615                 buffer = u.read(8192)
        616                 if len(buffer) == 0:
        617                     break
    

    ~\anaconda3\envs\mytorch\lib\http\client.py in read(self, amt)
        463                 # clip the read to the "end of response"
        464                 amt = self.length
    --> 465             s = self.fp.read(amt)
        466             if not s and amt:
        467                 # Ideally, we would raise IncompleteRead if the content-length
    

    ~\anaconda3\envs\mytorch\lib\socket.py in readinto(self, b)
        703         while True:
        704             try:
    --> 705                 return self._sock.recv_into(b)
        706             except timeout:
        707                 self._timeout_occurred = True
    

    ~\anaconda3\envs\mytorch\lib\ssl.py in recv_into(self, buffer, nbytes, flags)
       1272                   "non-zero flags not allowed in calls to recv_into() on %s" %
       1273                   self.__class__)
    -> 1274             return self.read(nbytes, buffer)
       1275         else:
       1276             return super().recv_into(buffer, nbytes, flags)
    

    ~\anaconda3\envs\mytorch\lib\ssl.py in read(self, len, buffer)
       1128         try:
       1129             if buffer is not None:
    -> 1130                 return self._sslobj.read(len, buffer)
       1131             else:
       1132                 return self._sslobj.read(len)
    

    KeyboardInterrupt: 



```python
torch.save(infer_model_0,'weights/b0_32_30.pt')
gc.collect()
torch.cuda.empty_cache()
```


```python
#0.76917 batch:16, epoch:30, LR: [10,20]
#0.73733 batch:8, epoch:30, LR: [9,18,27]
train_dataset_1 = CustomDataset(train_img_paths, train_labels, resize_transform(256,'train'))
train_loader_1 = DataLoader(train_dataset_1, batch_size = 32, shuffle=True, num_workers=0)

val_dataset_1 = CustomDataset(val_img_paths, val_labels, resize_transform(256,'test'))
val_loader_1 = DataLoader(val_dataset_1, batch_size=32, shuffle=False, num_workers=0)

model_1 = EfficientNet_B1()
optimizer_1 = torch.optim.Adam(params = model_1.parameters(), lr = CFG["LEARNING_RATE"])
scheduler_1 = lr_scheduler.MultiStepLR(optimizer_1, milestones=[10,20], gamma=0.5)
infer_model_1 = train(model_1, num_epoch, optimizer_1, train_loader_1, val_loader_1, scheduler_1, device)
```


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [1], Train Loss : [2.86433] Val Loss : [1.99244] Val F1 Score : [0.31057]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [2], Train Loss : [1.67252] Val Loss : [1.40388] Val F1 Score : [0.49893]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [3], Train Loss : [1.22175] Val Loss : [1.21196] Val F1 Score : [0.56305]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [4], Train Loss : [0.95499] Val Loss : [1.12642] Val F1 Score : [0.58965]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [5], Train Loss : [0.79811] Val Loss : [1.06909] Val F1 Score : [0.61608]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [6], Train Loss : [0.69485] Val Loss : [1.03311] Val F1 Score : [0.64123]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [7], Train Loss : [0.60306] Val Loss : [0.95537] Val F1 Score : [0.66869]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [8], Train Loss : [0.51950] Val Loss : [0.98454] Val F1 Score : [0.66737]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [9], Train Loss : [0.47799] Val Loss : [1.00903] Val F1 Score : [0.67572]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [10], Train Loss : [0.43844] Val Loss : [1.07134] Val F1 Score : [0.64893]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [11], Train Loss : [0.32390] Val Loss : [0.93084] Val F1 Score : [0.70103]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [12], Train Loss : [0.26255] Val Loss : [0.94579] Val F1 Score : [0.70830]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [13], Train Loss : [0.26448] Val Loss : [0.94592] Val F1 Score : [0.69609]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [14], Train Loss : [0.23658] Val Loss : [0.93292] Val F1 Score : [0.70796]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [15], Train Loss : [0.20293] Val Loss : [0.94833] Val F1 Score : [0.70960]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [16], Train Loss : [0.17906] Val Loss : [0.92146] Val F1 Score : [0.71044]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [17], Train Loss : [0.17759] Val Loss : [0.98141] Val F1 Score : [0.70534]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [18], Train Loss : [0.16463] Val Loss : [0.94423] Val F1 Score : [0.71908]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [19], Train Loss : [0.16075] Val Loss : [0.94291] Val F1 Score : [0.72070]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [20], Train Loss : [0.15083] Val Loss : [1.01654] Val F1 Score : [0.69910]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [21], Train Loss : [0.12245] Val Loss : [0.96108] Val F1 Score : [0.71936]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [22], Train Loss : [0.10746] Val Loss : [0.97904] Val F1 Score : [0.70149]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [23], Train Loss : [0.10572] Val Loss : [0.97803] Val F1 Score : [0.70993]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [24], Train Loss : [0.10125] Val Loss : [0.96736] Val F1 Score : [0.72027]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [25], Train Loss : [0.11162] Val Loss : [0.93451] Val F1 Score : [0.72733]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [26], Train Loss : [0.09709] Val Loss : [0.94990] Val F1 Score : [0.73513]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [27], Train Loss : [0.08806] Val Loss : [0.95070] Val F1 Score : [0.72676]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [28], Train Loss : [0.08901] Val Loss : [0.94382] Val F1 Score : [0.72633]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [29], Train Loss : [0.09446] Val Loss : [0.96864] Val F1 Score : [0.74452]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [30], Train Loss : [0.09075] Val Loss : [1.01470] Val F1 Score : [0.71360]
    


```python
torch.save(infer_model_1,'weights/b1_32_30.pt')
gc.collect()
torch.cuda.empty_cache()
```


```python
#0.78464 batch:16, epoch:30, lr:[10,20]
#0.77673 batch:8, epoch:30, LR: [9,18,27]

train_dataset_2 = CustomDataset(train_img_paths, train_labels, resize_transform(288,'train'))
train_loader_2 = DataLoader(train_dataset_2, batch_size = 32, shuffle=True, num_workers=0)

val_dataset_2 = CustomDataset(val_img_paths, val_labels, resize_transform(288,'test'))
val_loader_2 = DataLoader(val_dataset_2, batch_size=32, shuffle=False, num_workers=0)

model_2 = EfficientNet_B2()
optimizer_2 = torch.optim.Adam(params = model_2.parameters(), lr = CFG["LEARNING_RATE"])
scheduler_2 = lr_scheduler.MultiStepLR(optimizer_2, milestones=[10,20], gamma=0.5)
infer_model_2 = train(model_2, num_epoch, optimizer_2, train_loader_2, val_loader_2, scheduler_2, device)
```


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [1], Train Loss : [2.84091] Val Loss : [1.93657] Val F1 Score : [0.31513]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [2], Train Loss : [1.64862] Val Loss : [1.42578] Val F1 Score : [0.47108]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [3], Train Loss : [1.19200] Val Loss : [1.24434] Val F1 Score : [0.56843]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [4], Train Loss : [0.98051] Val Loss : [1.08336] Val F1 Score : [0.62683]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [5], Train Loss : [0.79707] Val Loss : [1.01485] Val F1 Score : [0.64318]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [6], Train Loss : [0.65105] Val Loss : [1.01123] Val F1 Score : [0.66262]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [7], Train Loss : [0.56123] Val Loss : [0.99846] Val F1 Score : [0.66938]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [8], Train Loss : [0.48988] Val Loss : [1.06386] Val F1 Score : [0.67323]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [9], Train Loss : [0.46574] Val Loss : [1.03603] Val F1 Score : [0.69025]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [10], Train Loss : [0.40618] Val Loss : [1.00115] Val F1 Score : [0.69509]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [11], Train Loss : [0.31095] Val Loss : [0.92462] Val F1 Score : [0.71284]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [12], Train Loss : [0.25567] Val Loss : [0.87565] Val F1 Score : [0.71627]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [13], Train Loss : [0.19559] Val Loss : [0.87135] Val F1 Score : [0.73637]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [14], Train Loss : [0.17889] Val Loss : [0.88735] Val F1 Score : [0.74174]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [15], Train Loss : [0.19776] Val Loss : [0.87272] Val F1 Score : [0.73257]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [16], Train Loss : [0.15405] Val Loss : [0.88796] Val F1 Score : [0.75005]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [17], Train Loss : [0.15535] Val Loss : [0.91011] Val F1 Score : [0.75121]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [18], Train Loss : [0.16650] Val Loss : [0.87328] Val F1 Score : [0.73690]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [19], Train Loss : [0.16779] Val Loss : [0.91495] Val F1 Score : [0.75160]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [20], Train Loss : [0.13167] Val Loss : [0.90711] Val F1 Score : [0.74207]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [21], Train Loss : [0.13070] Val Loss : [0.89215] Val F1 Score : [0.74640]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [22], Train Loss : [0.09696] Val Loss : [0.93472] Val F1 Score : [0.75965]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [23], Train Loss : [0.09565] Val Loss : [0.92335] Val F1 Score : [0.74133]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [24], Train Loss : [0.08867] Val Loss : [0.93646] Val F1 Score : [0.74623]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [25], Train Loss : [0.09513] Val Loss : [0.94538] Val F1 Score : [0.74109]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [26], Train Loss : [0.08413] Val Loss : [0.92614] Val F1 Score : [0.74130]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [27], Train Loss : [0.06902] Val Loss : [0.90424] Val F1 Score : [0.74870]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [28], Train Loss : [0.08088] Val Loss : [0.88133] Val F1 Score : [0.74694]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [29], Train Loss : [0.07385] Val Loss : [0.91839] Val F1 Score : [0.74788]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [30], Train Loss : [0.06524] Val Loss : [0.91463] Val F1 Score : [0.75428]
    


```python
torch.save(infer_model_2,'weights/b2_32_30.pt')
gc.collect()
torch.cuda.empty_cache()
```


```python
# 0.77395 batch:32, epoch:30, LR: [10,20]
# 0.77227 batch:16, epcoh:30, lr:[10,20]
# 0.76952 batch:8, epoch:30, LR: [9,18,27]

train_dataset_3 = CustomDataset(train_img_paths, train_labels, resize_transform(320,'train'))
train_loader_3 = DataLoader(train_dataset_3, batch_size = 32, shuffle=True, num_workers=0)

val_dataset_3 = CustomDataset(val_img_paths, val_labels, resize_transform(320,'test'))
val_loader_3 = DataLoader(val_dataset_3, batch_size=32, shuffle=False, num_workers=0)

model_3 = EfficientNet_B3()
optimizer_3 = torch.optim.Adam(params = model_3.parameters(), lr = CFG["LEARNING_RATE"])
scheduler_3 = lr_scheduler.MultiStepLR(optimizer_3, milestones=[10,20], gamma=0.5)
infer_model_3 = train(model_3, num_epoch, optimizer_3, train_loader_3, val_loader_3, scheduler_3, device)
```


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [1], Train Loss : [2.96602] Val Loss : [2.09809] Val F1 Score : [0.25765]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [2], Train Loss : [1.76417] Val Loss : [4.79204] Val F1 Score : [0.46767]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [3], Train Loss : [1.23941] Val Loss : [2.04095] Val F1 Score : [0.55870]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [4], Train Loss : [0.94343] Val Loss : [22.63151] Val F1 Score : [0.61632]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [5], Train Loss : [0.78273] Val Loss : [13.65165] Val F1 Score : [0.66445]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [6], Train Loss : [0.63187] Val Loss : [0.98746] Val F1 Score : [0.67272]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [7], Train Loss : [0.56337] Val Loss : [0.95072] Val F1 Score : [0.69470]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [8], Train Loss : [0.46253] Val Loss : [0.93798] Val F1 Score : [0.72060]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [9], Train Loss : [0.41566] Val Loss : [0.95672] Val F1 Score : [0.70088]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [10], Train Loss : [0.37844] Val Loss : [0.87937] Val F1 Score : [0.72910]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [11], Train Loss : [0.28186] Val Loss : [0.80918] Val F1 Score : [0.74541]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [12], Train Loss : [0.24858] Val Loss : [0.82082] Val F1 Score : [0.75996]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [13], Train Loss : [0.20853] Val Loss : [0.84702] Val F1 Score : [0.75718]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [14], Train Loss : [0.19173] Val Loss : [0.82337] Val F1 Score : [0.75086]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [15], Train Loss : [0.15849] Val Loss : [0.86349] Val F1 Score : [0.74895]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [16], Train Loss : [0.16425] Val Loss : [0.89469] Val F1 Score : [0.73866]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [17], Train Loss : [0.15805] Val Loss : [0.87677] Val F1 Score : [0.75369]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [18], Train Loss : [0.14943] Val Loss : [0.86080] Val F1 Score : [0.76789]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [19], Train Loss : [0.15045] Val Loss : [0.88198] Val F1 Score : [0.75696]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [20], Train Loss : [0.13075] Val Loss : [0.83852] Val F1 Score : [0.77286]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [21], Train Loss : [0.11035] Val Loss : [0.83203] Val F1 Score : [0.77791]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [22], Train Loss : [0.09658] Val Loss : [0.80675] Val F1 Score : [0.76831]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [23], Train Loss : [0.07447] Val Loss : [0.81352] Val F1 Score : [0.78098]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [24], Train Loss : [0.08322] Val Loss : [0.78007] Val F1 Score : [0.78147]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [25], Train Loss : [0.08468] Val Loss : [0.82178] Val F1 Score : [0.78063]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [26], Train Loss : [0.07550] Val Loss : [0.84984] Val F1 Score : [0.77687]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [27], Train Loss : [0.09197] Val Loss : [0.83565] Val F1 Score : [0.77364]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [28], Train Loss : [0.09117] Val Loss : [0.88094] Val F1 Score : [0.77142]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [29], Train Loss : [0.07765] Val Loss : [0.81785] Val F1 Score : [0.76842]
    


      0%|          | 0/148 [00:00<?, ?it/s]



      0%|          | 0/37 [00:00<?, ?it/s]


    Epoch [30], Train Loss : [0.07697] Val Loss : [1.13450] Val F1 Score : [0.76876]
    


```python
torch.save(infer_model_3,'weights/b3_32_30.pt')
gc.collect()
torch.cuda.empty_cache()
```


```python
#0.79137 batch:16, epoch:30, lr[10,20]
#0.79599 batch:8, epoch:30, LR: [9,18,27]

train_dataset_4 = CustomDataset(train_img_paths, train_labels, resize_transform(384,'train'))
train_loader_4 = DataLoader(train_dataset_4, batch_size = 16, shuffle=True, num_workers=0)

val_dataset_4 = CustomDataset(val_img_paths, val_labels, resize_transform(384,'test'))
val_loader_4 = DataLoader(val_dataset_4, batch_size=16, shuffle=False, num_workers=0)

model_4 = EfficientNet_B4()
optimizer_4 = torch.optim.Adam(params = model_4.parameters(), lr = CFG["LEARNING_RATE"])
scheduler_4 = lr_scheduler.MultiStepLR(optimizer_4, milestones=[10,20], gamma=0.5)
infer_model_4 = train(model_4, num_epoch, optimizer_4, train_loader_4, val_loader_4, scheduler_4, device)
```


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [1], Train Loss : [2.91452] Val Loss : [2.01377] Val F1 Score : [0.29155]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [2], Train Loss : [1.83046] Val Loss : [1.42001] Val F1 Score : [0.45443]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [3], Train Loss : [1.38424] Val Loss : [1.66429] Val F1 Score : [0.55680]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [4], Train Loss : [1.11449] Val Loss : [10.93012] Val F1 Score : [0.60079]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [5], Train Loss : [0.92050] Val Loss : [0.98696] Val F1 Score : [0.66394]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [6], Train Loss : [0.80853] Val Loss : [0.88997] Val F1 Score : [0.70736]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [7], Train Loss : [0.70011] Val Loss : [2.28833] Val F1 Score : [0.68705]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [8], Train Loss : [0.60581] Val Loss : [1.53085] Val F1 Score : [0.69880]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [9], Train Loss : [0.53688] Val Loss : [1.55519] Val F1 Score : [0.71713]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [10], Train Loss : [0.47595] Val Loss : [1.58916] Val F1 Score : [0.73828]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [11], Train Loss : [0.38231] Val Loss : [6.32857] Val F1 Score : [0.72827]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [12], Train Loss : [0.37177] Val Loss : [2.27956] Val F1 Score : [0.73956]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [13], Train Loss : [0.33208] Val Loss : [0.91502] Val F1 Score : [0.76212]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [14], Train Loss : [0.30703] Val Loss : [1.17400] Val F1 Score : [0.75043]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [15], Train Loss : [0.29056] Val Loss : [4.57718] Val F1 Score : [0.73297]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [16], Train Loss : [0.26246] Val Loss : [5.86595] Val F1 Score : [0.74171]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [17], Train Loss : [0.24770] Val Loss : [0.95076] Val F1 Score : [0.76121]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [18], Train Loss : [0.23292] Val Loss : [1.05648] Val F1 Score : [0.76255]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [19], Train Loss : [0.23622] Val Loss : [3.72793] Val F1 Score : [0.74958]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [20], Train Loss : [0.19691] Val Loss : [1.31184] Val F1 Score : [0.75902]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [21], Train Loss : [0.18796] Val Loss : [0.78035] Val F1 Score : [0.76742]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [22], Train Loss : [0.17296] Val Loss : [0.95592] Val F1 Score : [0.76110]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [23], Train Loss : [0.17477] Val Loss : [0.75100] Val F1 Score : [0.75959]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [24], Train Loss : [0.16227] Val Loss : [1.20974] Val F1 Score : [0.75511]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [25], Train Loss : [0.15522] Val Loss : [1.51613] Val F1 Score : [0.75224]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [26], Train Loss : [0.15120] Val Loss : [1.24624] Val F1 Score : [0.75991]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [27], Train Loss : [0.14973] Val Loss : [1.26375] Val F1 Score : [0.77462]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [28], Train Loss : [0.13176] Val Loss : [0.95489] Val F1 Score : [0.77805]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [29], Train Loss : [0.12939] Val Loss : [1.39810] Val F1 Score : [0.75337]
    


      0%|          | 0/296 [00:00<?, ?it/s]



      0%|          | 0/74 [00:00<?, ?it/s]


    Epoch [30], Train Loss : [0.12304] Val Loss : [0.89081] Val F1 Score : [0.77000]
    


```python
torch.save(infer_model_4,'weights/b4_16_30.pt')
gc.collect()
torch.cuda.empty_cache()
```


```python
#0.82004 batch:8, epoch : 30, lr:[9,18,27]
train_dataset_5 = CustomDataset(train_img_paths, train_labels, resize_transform(456,'train'))
train_loader_5 = DataLoader(train_dataset_5, batch_size = 8, shuffle=True, num_workers=0)

val_dataset_5 = CustomDataset(val_img_paths, val_labels, resize_transform(456,'test'))
val_loader_5 = DataLoader(val_dataset_5, batch_size = 8, shuffle=False, num_workers=0)

model_5 = EfficientNet_B5()
optimizer_5 = torch.optim.Adam(params = model_5.parameters(), lr = CFG["LEARNING_RATE"])
scheduler_5 = lr_scheduler.MultiStepLR(optimizer_5, milestones=[10,20], gamma=0.5)
infer_model_5 = train(model_5, num_epoch, optimizer_5, train_loader_5, val_loader_5, scheduler_5, device)
```


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [1], Train Loss : [2.51193] Val Loss : [1.97491] Val F1 Score : [0.30960]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [2], Train Loss : [1.64648] Val Loss : [1.43324] Val F1 Score : [0.46855]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [3], Train Loss : [1.32825] Val Loss : [1.23640] Val F1 Score : [0.58010]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [4], Train Loss : [1.14749] Val Loss : [1.13339] Val F1 Score : [0.61950]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [5], Train Loss : [1.01356] Val Loss : [1.17801] Val F1 Score : [0.63698]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [6], Train Loss : [0.92527] Val Loss : [1.04973] Val F1 Score : [0.67086]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [7], Train Loss : [0.78327] Val Loss : [1.04653] Val F1 Score : [0.68456]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [8], Train Loss : [0.76528] Val Loss : [1.00756] Val F1 Score : [0.67516]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [9], Train Loss : [0.70089] Val Loss : [1.04308] Val F1 Score : [0.68798]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [10], Train Loss : [0.64987] Val Loss : [1.12222] Val F1 Score : [0.68364]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [11], Train Loss : [0.44780] Val Loss : [0.84712] Val F1 Score : [0.73782]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [12], Train Loss : [0.36607] Val Loss : [0.89972] Val F1 Score : [0.74845]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [13], Train Loss : [0.34390] Val Loss : [0.82024] Val F1 Score : [0.77353]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [14], Train Loss : [0.31029] Val Loss : [0.84894] Val F1 Score : [0.78619]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [15], Train Loss : [0.27588] Val Loss : [0.90589] Val F1 Score : [0.75371]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [16], Train Loss : [0.27717] Val Loss : [0.85182] Val F1 Score : [0.77368]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [17], Train Loss : [0.25863] Val Loss : [0.90642] Val F1 Score : [0.75616]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [18], Train Loss : [0.24753] Val Loss : [0.89566] Val F1 Score : [0.76424]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [19], Train Loss : [0.25179] Val Loss : [0.89594] Val F1 Score : [0.76241]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [20], Train Loss : [0.24201] Val Loss : [0.90138] Val F1 Score : [0.74203]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [21], Train Loss : [0.18337] Val Loss : [0.82500] Val F1 Score : [0.78208]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [22], Train Loss : [0.14128] Val Loss : [0.81791] Val F1 Score : [0.79089]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [23], Train Loss : [0.12550] Val Loss : [0.80201] Val F1 Score : [0.79456]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [24], Train Loss : [0.12944] Val Loss : [0.83640] Val F1 Score : [0.79615]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [25], Train Loss : [0.12516] Val Loss : [0.82149] Val F1 Score : [0.79930]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [26], Train Loss : [0.11178] Val Loss : [0.81104] Val F1 Score : [0.79597]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [27], Train Loss : [0.11839] Val Loss : [0.82280] Val F1 Score : [0.79565]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [28], Train Loss : [0.10380] Val Loss : [0.82293] Val F1 Score : [0.80303]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [29], Train Loss : [0.11044] Val Loss : [0.87055] Val F1 Score : [0.79833]
    


      0%|          | 0/591 [00:00<?, ?it/s]



      0%|          | 0/148 [00:00<?, ?it/s]


    Epoch [30], Train Loss : [0.09429] Val Loss : [0.84395] Val F1 Score : [0.79083]
    


```python
torch.save(infer_model_5,'weights/b5_8_30.pt')
gc.collect()
torch.cuda.empty_cache()
```


```python
#0.80073 batch:4, epoch:30, lr:[8,16,24]
train_dataset_6 = CustomDataset(train_img_paths, train_labels, resize_transform(528,'train'))
train_loader_6 = DataLoader(train_dataset_6, batch_size = 4, shuffle=True, num_workers=0)

val_dataset_6 = CustomDataset(val_img_paths, val_labels, resize_transform(528,'test'))
val_loader_6 = DataLoader(val_dataset_6, batch_size=4, shuffle=False, num_workers=0)

model_6 = EfficientNet_B6()
optimizer_6 = torch.optim.Adam(params = model_6.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = lr_scheduler.MultiStepLR(optimizer_6, milestones=[10,20], gamma=0.5)
infer_model_6 = train(model_6,num_epoch, optimizer_6, train_loader_6, val_loader_6, scheduler, device)
```


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [1], Train Loss : [2.75712] Val Loss : [2.27502] Val F1 Score : [0.26730]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [2], Train Loss : [2.07096] Val Loss : [1.73975] Val F1 Score : [0.39114]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [3], Train Loss : [1.78116] Val Loss : [1.79645] Val F1 Score : [0.46726]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [4], Train Loss : [1.59450] Val Loss : [1.37452] Val F1 Score : [0.55316]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [5], Train Loss : [1.44005] Val Loss : [1.50434] Val F1 Score : [0.56682]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [6], Train Loss : [1.32243] Val Loss : [1.26569] Val F1 Score : [0.61335]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [7], Train Loss : [1.25429] Val Loss : [1.67486] Val F1 Score : [0.59811]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [8], Train Loss : [1.13941] Val Loss : [1.33604] Val F1 Score : [0.63580]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [9], Train Loss : [1.04273] Val Loss : [1.36965] Val F1 Score : [0.63320]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [10], Train Loss : [1.03504] Val Loss : [1.12011] Val F1 Score : [0.65805]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [11], Train Loss : [0.75325] Val Loss : [1.08717] Val F1 Score : [0.70159]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [12], Train Loss : [0.62935] Val Loss : [1.12726] Val F1 Score : [0.72934]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [13], Train Loss : [0.62206] Val Loss : [1.17660] Val F1 Score : [0.71835]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [14], Train Loss : [0.57592] Val Loss : [1.08389] Val F1 Score : [0.71349]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [15], Train Loss : [0.52196] Val Loss : [1.11982] Val F1 Score : [0.73091]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [16], Train Loss : [0.51311] Val Loss : [1.07925] Val F1 Score : [0.73683]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [17], Train Loss : [0.49379] Val Loss : [1.02476] Val F1 Score : [0.73703]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [18], Train Loss : [0.44150] Val Loss : [1.05356] Val F1 Score : [0.73342]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [19], Train Loss : [0.45648] Val Loss : [1.16292] Val F1 Score : [0.71722]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [20], Train Loss : [0.40129] Val Loss : [1.23133] Val F1 Score : [0.71932]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [21], Train Loss : [0.32776] Val Loss : [1.00287] Val F1 Score : [0.75250]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [22], Train Loss : [0.26870] Val Loss : [1.06099] Val F1 Score : [0.75319]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [23], Train Loss : [0.24317] Val Loss : [1.04907] Val F1 Score : [0.75398]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [24], Train Loss : [0.23379] Val Loss : [1.04655] Val F1 Score : [0.74993]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [25], Train Loss : [0.21799] Val Loss : [0.99212] Val F1 Score : [0.75734]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [26], Train Loss : [0.21675] Val Loss : [1.04751] Val F1 Score : [0.74955]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [27], Train Loss : [0.21946] Val Loss : [1.08012] Val F1 Score : [0.75314]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [28], Train Loss : [0.20736] Val Loss : [1.05106] Val F1 Score : [0.77193]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [29], Train Loss : [0.18628] Val Loss : [1.16049] Val F1 Score : [0.76815]
    


      0%|          | 0/1182 [00:00<?, ?it/s]



      0%|          | 0/296 [00:00<?, ?it/s]


    Epoch [30], Train Loss : [0.17886] Val Loss : [1.06788] Val F1 Score : [0.77537]
    


```python
torch.save(infer_model_6,'weights/b6_4_30.pt')
gc.collect()
torch.cuda.empty_cache()
```

### 5. Test PyTorch Model


```python
test_df = pd.read_csv(f'{data_dir}/test.csv')
test_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>img_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_00000</td>
      <td>./test/TEST_00000.jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_00001</td>
      <td>./test/TEST_00001.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_00002</td>
      <td>./test/TEST_00002.jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TEST_00003</td>
      <td>./test/TEST_00003.jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TEST_00004</td>
      <td>./test/TEST_00004.jpg</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_img_paths = get_data(test_df, infer=True)
```


```python
test_dataset_1 = CustomDataset(test_img_paths, None, resize_transform(256,'test'))
test_loader_1 = DataLoader(test_dataset_1, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

test_dataset_2 = CustomDataset(test_img_paths, None, resize_transform(288,'test'))
test_loader_2 = DataLoader(test_dataset_2, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
                               
test_dataset_3 = CustomDataset(test_img_paths, None, resize_transform(320,'test'))
test_loader_3 = DataLoader(test_dataset_3, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
                               
test_dataset_4 = CustomDataset(test_img_paths, None, resize_transform(384,'test'))
test_loader_4 = DataLoader(test_dataset_4, batch_size=16, shuffle=False, num_workers=0)
                               
test_dataset_5 = CustomDataset(test_img_paths, None, resize_transform(456,'test'))
test_loader_5 = DataLoader(test_dataset_5, batch_size=16, shuffle=False, num_workers=0)

test_dataset_6 = CustomDataset(test_img_paths, None, resize_transform(528,'test'))
test_loader_6 = DataLoader(test_dataset_6, batch_size=8, shuffle=False, num_workers=0)

                               
```

# Inference Test


```python
def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    
    model_preds = []
    
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.float().to(device)
            
            model_pred = model(img)
            model_preds += model_pred.detach().cpu().numpy().tolist()
            
    
    print('Done.')
    return model_preds
```


```python
#preds = inference(infer_model_5, test_loader_5, device)
```

## TF model

### 1.Import Library


```python
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import math
import scipy as sp
import PIL
import logging
import warnings
import matplotlib.style as style
import seaborn as sns
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# Tensorflow
from tensorflow.keras import models, layers, Model, regularizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout, ZeroPadding2D 
from sklearn import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau,  EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB6, ResNet50V2, EfficientNetB0
#from keras_tuner.tuners import RandomSearch
from tensorflow.keras.utils import Sequence
#import scikitplot as skplt
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
#from albumentations import Compose, HorizontalFlip, CLAHE, HueSaturationValue,RandomBrightness, RandomContrast, RandomGamma,ToFloat, ShiftScaleRotate
import albumentations as A

```

### 2.Make Dataset


```python
class CustomTFDataset(Sequence):
    def __init__(self, x_set, y_set, batch_size, augmentations, shuffle = True):
        self.y = y_set
        self.x = x_set
        self.batch_size = batch_size
        self.augment = augmentations
        self.indexes = np.arange(self.x.shape[0])
        self.shuffle = shuffle
        self.on_epoch_end()
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    # 지정 배치 크기만큼 데이터를 로드합니다.
    def __getitem__(self, idx):
        inds = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        img_path = self.x[inds]
        batch_x = [cv2.cvtColor(cv2.imread(f"{data_dir}/{x.split('/')[-2]}/{x.split('/')[-1]}"), cv2.COLOR_BGR2RGB)for x in img_path]
        #batch_x = self.x[inds]
        batch_y = self.y[inds]
        
        # augmentation을 적용해서 numpy array에 stack합니다.
        return np.stack([
            self.augment(image=x)["image"] for x in batch_x
        ], axis=0), np.array(batch_y)
```


```python
input_size = [224, 240, 260, 300 ,380, 456, 528, 600]
```


```python
train_transform = [A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(limit= 90),
                              #A.ToGray(p=0.2), A.RandomBrightnessContrast(0.2, 0.2, p=0.2),
                              A.RandomResizedCrop(X, X, (0.08, 1.0)),
                              A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                   ])
                   for X in input_size]

test_transform = [A.Compose([A.Resize(X,X),A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)])
                  for X in input_size]
```


```python
one_train_labels = tf.keras.utils.to_categorical(train_labels)
one_val_labels = tf.keras.utils.to_categorical(val_labels)
```


```python
train_gen = [CustomTFDataset(train_img_paths, one_train_labels, 16 if x < 5 else 4 , TT, shuffle =True)
            for x ,TT in enumerate (train_transform)]
val_gen = [CustomTFDataset(val_img_paths, one_val_labels, 16 if x < 5 else 4, TT, shuffle =False)
          for x, TT in enumerate (test_transform)]
```

### 3.Make TF Model


```python
ensemble_model = []
```


```python
def ada(e): 
    return tf.keras.optimizers.Adam(learning_rate=e)
```


```python
#0
base_model = tf.keras.applications.EfficientNetB0(include_top=False,
                   input_shape=(224,224,3),
                   pooling='avg',classes=1000,
                   weights='imagenet')
X = Flatten()(base_model.output)
#X = Dense(1280, activation = 'relu')(X)
#X = BatchNormalization()(X)
#X = Dropout(0.2)(X)
X = Dense(500, activation = 'relu')(X)
X = BatchNormalization()(X)
X = Dropout(0.2)(X)
X = Dense(50, activation='softmax')(base_model.output)
image_model = Model(inputs = base_model.input, outputs = X)
image_model.compile(optimizer=ada(3e-4), loss='categorical_crossentropy', metrics=['accuracy'])
ensemble_model.append(image_model)
```


```python
#1
base_model = tf.keras.applications.EfficientNetB1(include_top=False,
                   input_shape=(240,240,3),
                   pooling='avg',classes=1000,
                   weights='imagenet')
X = Flatten()(base_model.output)
#X = Dense(1280, activation = 'relu')(X)
#X = BatchNormalization()(X)
#X = Dropout(0.2)(X)
X = Dense(500, activation = 'relu')(X)
X = BatchNormalization()(X)
X = Dropout(0.2)(X)
X = Dense(50, activation='softmax')(X)
image_model = Model(inputs = base_model.input, outputs = X)
image_model.compile(optimizer=ada(3e-4), loss='categorical_crossentropy', metrics=['accuracy'])
ensemble_model.append(image_model)
```


```python
#2
base_model = tf.keras.applications.EfficientNetB2(include_top=False,
                   input_shape=(260,260,3),
                   pooling='avg',classes=1000,
                   weights='imagenet')
X = Flatten()(base_model.output)
#X = Dense(1408, activation = 'relu')(X)
#X = BatchNormalization()(X)
#X = Dropout(0.2)(X)
X = Dense(500, activation = 'relu')(X)
X = BatchNormalization()(X)
X = Dropout(0.2)(X)
X = Dense(50, activation='softmax')(X)
image_model = Model(inputs = base_model.input, outputs = X)
image_model.compile(optimizer=ada(3e-4), loss='categorical_crossentropy', metrics=['accuracy'])
ensemble_model.append(image_model)
```


```python
#3
base_model = tf.keras.applications.EfficientNetB3(include_top=False,
                   input_shape=(300,300,3),
                   pooling='avg',classes=1000,
                   weights='imagenet')
X = Flatten()(base_model.output)
#X = Dense(1536, activation = 'relu')(X)
#X = BatchNormalization()(X)
#X = Dropout(0.2)(X)
X = Dense(500, activation = 'relu')(X)
X = BatchNormalization()(X)
X = Dropout(0.2)(X)
X = Dense(50, activation='softmax')(X)
image_model = Model(inputs = base_model.input, outputs = X)
image_model.compile(optimizer=ada(3e-4), loss='categorical_crossentropy', metrics=['accuracy'])
ensemble_model.append(image_model)
```


```python
#4
base_model = tf.keras.applications.EfficientNetB4(include_top=False,
                   input_shape=(380,380,3),
                   pooling='avg',classes=1000,
                   weights='imagenet')
X = Flatten()(base_model.output)
#X = Dense(1792, activation = 'relu')(X)
#X = BatchNormalization()(X)
#X = Dropout(0.2)(X)
X = Dense(500, activation = 'relu')(X)
X = BatchNormalization()(X)
X = Dropout(0.2)(X)
X = Dense(50, activation='softmax')(X)
image_model = Model(inputs = base_model.input, outputs = X)
image_model.compile(optimizer=ada(3e-4), loss='categorical_crossentropy', metrics=['accuracy'])
ensemble_model.append(image_model)
```


```python
#5
base_model = tf.keras.applications.EfficientNetB5(include_top=False,
                   input_shape=(456,456,3),
                   pooling='avg',classes=1000,
                   weights='imagenet')
X = Flatten()(base_model.output)
#X = Dense(2048, activation = 'relu')(X)
#X = BatchNormalization()(X)
#X = Dropout(0.2)(X)
X = Dense(500, activation = 'relu')(X)
X = BatchNormalization()(X)
X = Dropout(0.2)(X)
X = Dense(50, activation='softmax')(X)
image_model = Model(inputs = base_model.input, outputs = X)
image_model.compile(optimizer=ada(3e-4), loss='categorical_crossentropy', metrics=['accuracy'])
ensemble_model.append(image_model)
```


```python
#6
base_model = tf.keras.applications.EfficientNetB6(include_top=False,
                   input_shape=(528,528,3),
                   pooling='avg',classes=1000,
                   weights='imagenet')
X = Flatten()(base_model.output)
#X = Dense(2304, activation = 'relu')(X)
#X = BatchNormalization()(X)
#X = Dropout(0.2)(X)
X = Dense(500, activation = 'relu')(X)
X = BatchNormalization()(X)
X = Dropout(0.2)(X)
X = Dense(50, activation='softmax')(X)
image_model = Model(inputs = base_model.input, outputs = X)
image_model.compile(optimizer=ada(3e-4), loss='categorical_crossentropy', metrics=['accuracy'])
ensemble_model.append(image_model)
```

### 4.Train TF model


```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
```

    [name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 15786879934872305681
    xla_global_id: -1
    , name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 22720544768
    locality {
      bus_id: 1
      links {
      }
    }
    incarnation: 14073362446480609761
    physical_device_desc: "device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:01:00.0, compute capability: 8.6"
    xla_global_id: 416903419
    ]
    


```python
def ada(e): 
    return tf.keras.optimizers.Adam(learning_rate=e)
```


```python
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
def mcp_save(name):
     return ModelCheckpoint('weights/'+name +'model.h5', save_best_only=True, monitor='val_accuracy', mode='max',save_weights_only=True)
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=1e-4, mode='min')
```

    WARNING:tensorflow:`epsilon` argument is deprecated and will be removed, use `min_delta` instead.
    


```python
history= [train_model.fit(train_gen[name], epochs=30, validation_data=val_gen[name],
                          callbacks=[earlyStopping, mcp_save(str(name)), reduce_lr_loss])for name, train_model in enumerate (ensemble_model)]
```

    Epoch 1/30
    296/296 [==============================] - 61s 205ms/step - loss: 0.8231 - accuracy: 0.7570 - val_loss: 5.2849 - val_accuracy: 0.1378 - lr: 3.0000e-04
    Epoch 2/30
     24/296 [=>............................] - ETA: 46s - loss: 0.8307 - accuracy: 0.7526


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_30064\304949652.py in <cell line: 1>()
    ----> 1 history= [train_model.fit(train_gen[name], epochs=30, validation_data=val_gen[name],
          2                           callbacks=[earlyStopping, mcp_save(str(name)), reduce_lr_loss])for name, train_model in enumerate (ensemble_model)]
    

    ~\AppData\Local\Temp\ipykernel_30064\304949652.py in <listcomp>(.0)
    ----> 1 history= [train_model.fit(train_gen[name], epochs=30, validation_data=val_gen[name],
          2                           callbacks=[earlyStopping, mcp_save(str(name)), reduce_lr_loss])for name, train_model in enumerate (ensemble_model)]
    

    ~\anaconda3\envs\mytorch\lib\site-packages\keras\utils\traceback_utils.py in error_handler(*args, **kwargs)
         63         filtered_tb = None
         64         try:
    ---> 65             return fn(*args, **kwargs)
         66         except Exception as e:
         67             filtered_tb = _process_traceback_frames(e.__traceback__)
    

    ~\anaconda3\envs\mytorch\lib\site-packages\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
       1562                         ):
       1563                             callbacks.on_train_batch_begin(step)
    -> 1564                             tmp_logs = self.train_function(iterator)
       1565                             if data_handler.should_sync:
       1566                                 context.async_wait()
    

    ~\anaconda3\envs\mytorch\lib\site-packages\tensorflow\python\util\traceback_utils.py in error_handler(*args, **kwargs)
        148     filtered_tb = None
        149     try:
    --> 150       return fn(*args, **kwargs)
        151     except Exception as e:
        152       filtered_tb = _process_traceback_frames(e.__traceback__)
    

    ~\anaconda3\envs\mytorch\lib\site-packages\tensorflow\python\eager\def_function.py in __call__(self, *args, **kwds)
        913 
        914       with OptionalXlaContext(self._jit_compile):
    --> 915         result = self._call(*args, **kwds)
        916 
        917       new_tracing_count = self.experimental_get_tracing_count()
    

    ~\anaconda3\envs\mytorch\lib\site-packages\tensorflow\python\eager\def_function.py in _call(self, *args, **kwds)
        945       # In this case we have created variables on the first call, so we run the
        946       # defunned version which is guaranteed to never create variables.
    --> 947       return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
        948     elif self._stateful_fn is not None:
        949       # Release the lock early so that multiple threads can perform the call
    

    ~\anaconda3\envs\mytorch\lib\site-packages\tensorflow\python\eager\function.py in __call__(self, *args, **kwargs)
       2494       (graph_function,
       2495        filtered_flat_args) = self._maybe_define_function(args, kwargs)
    -> 2496     return graph_function._call_flat(
       2497         filtered_flat_args, captured_inputs=graph_function.captured_inputs)  # pylint: disable=protected-access
       2498 
    

    ~\anaconda3\envs\mytorch\lib\site-packages\tensorflow\python\eager\function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
       1860         and executing_eagerly):
       1861       # No tape is watching; skip to running the function.
    -> 1862       return self._build_call_outputs(self._inference_function.call(
       1863           ctx, args, cancellation_manager=cancellation_manager))
       1864     forward_backward = self._select_forward_and_backward_functions(
    

    ~\anaconda3\envs\mytorch\lib\site-packages\tensorflow\python\eager\function.py in call(self, ctx, args, cancellation_manager)
        497       with _InterpolateFunctionError(self):
        498         if cancellation_manager is None:
    --> 499           outputs = execute.execute(
        500               str(self.signature.name),
        501               num_outputs=self._num_outputs,
    

    ~\anaconda3\envs\mytorch\lib\site-packages\tensorflow\python\eager\execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         52   try:
         53     ctx.ensure_initialized()
    ---> 54     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
         55                                         inputs, attrs, num_outputs)
         56   except core._NotOkStatusException as e:
    

    KeyboardInterrupt: 



```python
scores = [ensemble_model[x].evaluate_generator(val_gen[x], verbose=0) for x in range(7)]
```


```python
for x in range(7):
    print("%s: %.2f%%" %(ensemble_model[x].metrics_names[1], scores[x][1]*100))
```


```python
fig1 = plt.gcf()
for i in range(7):
    plt.plot(history[i].history['loss'])
    plt.plot(history[i].history['val_loss'])
    plt.axis(ymin=0.1,ymax=10)
    plt.axis(xmax = 30)
    plt.grid()
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])
plt.show()
```

### 5.Test TF model


```python
for x in range(7):
    print(ensemble_model[x])
    ensemble_model[x].load_weights(filepath='weights/'+str(x)+'model.h5')
```

    <keras.engine.functional.Functional object at 0x000001812D30B2E0>
    <keras.engine.functional.Functional object at 0x00000181472587C0>
    <keras.engine.functional.Functional object at 0x000001812D336320>
    <keras.engine.functional.Functional object at 0x0000018164CAB880>
    <keras.engine.functional.Functional object at 0x0000018164A9DEA0>
    <keras.engine.functional.Functional object at 0x0000018164CABE50>
    <keras.engine.functional.Functional object at 0x0000018164AD8B20>
    


```python
test_transform = [A.Compose([A.Resize(X,X),A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)])
                  for X in input_size]
```


```python
test_img_paths = get_data(test_df, infer=True)
test_gen = [CustomTFDataset(test_img_paths, test_img_paths, 16, x, shuffle =False)for x in test_transform]
```


```python
test_gen[6].batch_size = 8
```


```python
gc.collect()
torch.cuda.empty_cache()
```


```python
Test_Predic = [ensemble_model[x].predict(test_gen[x])for x in range(1,7)]
```

    792/792 [==============================] - 43s 52ms/step
    792/792 [==============================] - 43s 53ms/step
    792/792 [==============================] - 48s 59ms/step
    792/792 [==============================] - 56s 69ms/step
    792/792 [==============================] - 93s 115ms/step
    1584/1584 [==============================] - 170s 105ms/step
    

## Ensemble PyTorch and TensorFlow


```python
torch_model_1 = torch.load('weights/b1_32_30.pt')
torch_model_2 = torch.load('weights/b2_32_30.pt')
torch_model_3 = torch.load('weights/b3_32_30.pt')
torch_model_4 = torch.load('weights/b4_16_30.pt')
torch_model_5 = torch.load('weights/b5_8_30.pt')
torch_model_6 = torch.load('weights/b6_4_30.pt')
```


```python
model_list = [torch_model_1,torch_model_2, torch_model_3,torch_model_4,torch_model_5,torch_model_6]
test_dataset_list = [ test_loader_1,test_loader_2, test_loader_3, test_loader_4, test_loader_5, test_loader_6]

pred = []
for i in range(len(model_list)):
    pred += [inference(model_list[i],test_dataset_list[i],device)]

```


      0%|          | 0/396 [00:00<?, ?it/s]


    Done.
    


      0%|          | 0/396 [00:00<?, ?it/s]


    Done.
    


      0%|          | 0/396 [00:00<?, ?it/s]


    Done.
    


      0%|          | 0/792 [00:00<?, ?it/s]


    Done.
    


      0%|          | 0/792 [00:00<?, ?it/s]


    Done.
    


      0%|          | 0/1584 [00:00<?, ?it/s]


    Done.
    


```python
pred_tf = Test_Predic

result = []
for i_0,i_1,i_2,i_3,i_4,i_5,i_tf0,i_tf1,i_tf2,i_tf3,i_tf4,i_tf5 in zip(pred[0],pred[1],pred[2],pred[3],pred[4],pred[5],
                                        pred_tf[0],pred_tf[1],pred_tf[2],pred_tf[3],pred_tf[4],pred_tf[5]):#,pred_tf[6]):
    i_0 = F.softmax(torch.Tensor(i_0))
    i_1 = F.softmax(torch.Tensor(i_1))
    i_2 = F.softmax(torch.Tensor(i_2))
    i_3 = F.softmax(torch.Tensor(i_3))
    i_4 = F.softmax(torch.Tensor(i_4))
    i_5 = F.softmax(torch.Tensor(i_5))

    i_tf0 = torch.Tensor(i_tf0)
    i_tf1 = torch.Tensor(i_tf1)
    i_tf2 = torch.Tensor(i_tf2)
    i_tf3 = torch.Tensor(i_tf3)
    i_tf4 = torch.Tensor(i_tf4)
    i_tf5 = torch.Tensor(i_tf5)
    #i_tf6 = torch.Tensor(i_tf6)

    i = i_0 + i_1 + i_2 + i_3 + i_4 +i_5 +i_tf0 +i_tf1 +i_tf2 +i_tf3 +i_tf4 +i_tf5 #+i_tf6

    i_arg = i.argmax()
    result.append(i_arg.item())
print(result)
```

    [10, 2, 29, 0, 10, 33, 33, 48, 11, 40, 32, 43, 48, 10, 45, 33, 16, 33, 18, 38, 38, 22, 38, 15, 33, 38, 48, 48, 44, 33, 16, 32, 9, 21, 48, 10, 35, 4, 21, 46, 18, 45, 10, 10, 44, 3, 9, 42, 36, 10, 5, 47, 13, 10, 3, 2, 41, 13, 33, 29, 34, 35, 2, 15, 42, 10, 43, 42, 10, 46, 10, 0, 45, 32, 38, 15, 10, 30, 0, 48, 36, 24, 10, 7, 48, 33, 46, 40, 23, 43, 27, 1, 5, 27, 40, 5, 33, 20, 38, 1, 35, 22, 21, 18, 45, 46, 1, 48, 1, 0, 38, 38, 11, 21, 26, 48, 8, 33, 46, 17, 4, 2, 3, 42, 43, 37, 29, 48, 9, 42, 1, 43, 39, 40, 9, 29, 33, 1, 32, 46, 38, 9, 10, 29, 1, 10, 47, 37, 48, 20, 1, 37, 10, 10, 43, 46, 29, 16, 38, 41, 3, 15, 48, 38, 33, 34, 0, 38, 21, 48, 48, 11, 30, 23, 44, 35, 20, 0, 0, 42, 38, 38, 39, 42, 15, 10, 10, 0, 33, 10, 8, 35, 21, 15, 18, 30, 40, 40, 1, 10, 30, 0, 30, 15, 1, 10, 1, 21, 4, 38, 0, 48, 10, 0, 35, 19, 6, 24, 10, 10, 10, 2, 15, 48, 11, 10, 38, 1, 30, 33, 47, 32, 16, 24, 38, 35, 19, 4, 2, 21, 6, 35, 10, 7, 15, 2, 10, 39, 8, 4, 15, 0, 0, 45, 33, 24, 48, 45, 48, 46, 44, 30, 19, 19, 48, 36, 48, 1, 41, 10, 33, 30, 8, 9, 28, 1, 42, 0, 2, 12, 0, 27, 36, 4, 33, 46, 21, 29, 19, 6, 48, 48, 36, 35, 4, 48, 2, 1, 11, 32, 42, 40, 16, 43, 12, 26, 5, 2, 38, 37, 10, 32, 4, 4, 15, 1, 40, 1, 9, 13, 49, 48, 1, 10, 35, 48, 42, 13, 37, 10, 10, 41, 4, 19, 37, 9, 10, 5, 1, 0, 48, 40, 21, 48, 33, 35, 15, 48, 18, 43, 26, 15, 43, 18, 2, 44, 0, 48, 48, 41, 41, 2, 39, 26, 38, 21, 5, 16, 32, 45, 1, 48, 4, 48, 4, 15, 42, 35, 46, 4, 16, 38, 40, 8, 36, 43, 44, 46, 44, 33, 38, 35, 10, 10, 2, 40, 35, 32, 1, 1, 10, 28, 1, 30, 1, 45, 48, 33, 10, 19, 18, 48, 33, 48, 10, 33, 9, 9, 1, 5, 21, 25, 48, 48, 0, 46, 45, 9, 46, 44, 2, 43, 0, 42, 37, 30, 38, 18, 32, 4, 10, 48, 27, 19, 33, 32, 8, 10, 0, 11, 15, 47, 37, 32, 2, 28, 15, 30, 45, 10, 28, 19, 40, 48, 35, 15, 30, 11, 33, 21, 0, 44, 19, 2, 30, 37, 33, 16, 48, 0, 37, 5, 33, 26, 38, 39, 49, 5, 28, 48, 24, 42, 43, 38, 38, 22, 43, 10, 1, 38, 33, 33, 27, 24, 38, 9, 48, 48, 36, 36, 48, 32, 10, 15, 33, 4, 33, 48, 10, 1, 1, 42, 18, 21, 19, 28, 36, 32, 48, 26, 16, 38, 0, 48, 28, 12, 46, 1, 42, 48, 30, 18, 43, 31, 33, 0, 43, 31, 48, 6, 33, 18, 46, 48, 43, 48, 9, 38, 22, 48, 33, 10, 19, 9, 35, 35, 21, 13, 48, 45, 1, 42, 45, 27, 10, 32, 48, 29, 30, 0, 0, 30, 48, 46, 48, 48, 36, 48, 0, 4, 2, 44, 1, 15, 41, 4, 2, 45, 45, 5, 30, 15, 9, 22, 9, 2, 9, 15, 28, 47, 9, 33, 10, 38, 44, 17, 48, 18, 16, 21, 15, 48, 19, 10, 48, 10, 46, 30, 38, 38, 38, 34, 36, 4, 32, 48, 3, 48, 16, 37, 38, 46, 49, 10, 1, 48, 10, 3, 44, 48, 16, 45, 12, 9, 48, 7, 30, 18, 1, 1, 0, 10, 46, 42, 28, 21, 32, 17, 21, 38, 35, 18, 2, 10, 1, 33, 33, 19, 10, 33, 36, 38, 24, 35, 42, 0, 7, 48, 48, 48, 30, 0, 1, 6, 8, 43, 16, 35, 48, 26, 21, 30, 1, 30, 48, 19, 15, 1, 38, 10, 48, 48, 0, 48, 15, 37, 44, 48, 31, 9, 35, 38, 4, 10, 35, 42, 46, 38, 42, 2, 27, 2, 30, 18, 48, 10, 32, 33, 41, 20, 38, 18, 9, 48, 10, 21, 43, 11, 10, 48, 38, 8, 37, 38, 1, 48, 27, 10, 38, 10, 39, 30, 45, 10, 11, 20, 42, 24, 43, 2, 49, 30, 33, 10, 48, 37, 15, 15, 10, 19, 48, 48, 30, 2, 1, 3, 38, 10, 0, 48, 13, 3, 37, 48, 4, 10, 29, 13, 8, 2, 33, 37, 6, 48, 32, 2, 48, 8, 45, 4, 29, 40, 28, 48, 29, 46, 33, 0, 43, 48, 35, 1, 48, 40, 24, 0, 10, 10, 15, 46, 1, 1, 33, 6, 44, 10, 48, 33, 46, 38, 48, 10, 10, 26, 46, 33, 18, 0, 0, 48, 38, 10, 21, 42, 24, 16, 35, 0, 1, 3, 25, 7, 0, 10, 41, 6, 33, 10, 42, 39, 48, 40, 10, 47, 32, 15, 37, 24, 48, 1, 0, 10, 27, 18, 19, 22, 38, 21, 27, 35, 46, 21, 10, 48, 43, 10, 40, 35, 38, 10, 48, 27, 10, 48, 0, 45, 44, 28, 46, 10, 20, 40, 10, 33, 35, 22, 41, 33, 27, 48, 48, 15, 38, 19, 42, 48, 46, 43, 0, 47, 48, 29, 14, 15, 12, 33, 48, 18, 48, 18, 48, 15, 12, 44, 15, 35, 46, 5, 5, 21, 10, 28, 43, 10, 3, 0, 16, 19, 46, 10, 35, 37, 9, 10, 36, 23, 6, 33, 10, 19, 10, 46, 33, 34, 10, 22, 48, 42, 48, 48, 0, 44, 32, 35, 10, 33, 2, 19, 48, 32, 32, 48, 10, 33, 44, 38, 1, 35, 12, 41, 10, 43, 32, 46, 0, 48, 27, 21, 4, 35, 5, 0, 19, 46, 28, 27, 18, 41, 40, 32, 32, 38, 15, 48, 32, 1, 29, 21, 36, 10, 10, 10, 44, 10, 10, 42, 35, 10, 48, 42, 21, 11, 28, 13, 38, 42, 21, 15, 48, 48, 33, 13, 10, 10, 23, 0, 30, 3, 3, 13, 29, 1, 19, 48, 10, 30, 21, 22, 24, 1, 41, 16, 33, 46, 9, 21, 9, 21, 46, 43, 23, 43, 48, 48, 40, 48, 1, 48, 46, 45, 1, 45, 30, 33, 26, 44, 15, 49, 33, 4, 21, 10, 36, 48, 48, 18, 7, 30, 48, 10, 38, 10, 21, 2, 19, 38, 49, 48, 38, 35, 41, 43, 38, 38, 21, 43, 0, 29, 19, 8, 28, 10, 2, 28, 9, 44, 46, 3, 10, 4, 46, 35, 13, 48, 15, 10, 2, 30, 4, 8, 2, 28, 2, 33, 29, 2, 45, 9, 35, 4, 27, 36, 30, 34, 34, 35, 9, 38, 28, 10, 33, 48, 42, 37, 33, 46, 48, 9, 39, 22, 48, 46, 42, 38, 15, 46, 38, 46, 14, 32, 5, 16, 36, 49, 48, 10, 30, 4, 38, 48, 0, 43, 37, 2, 10, 10, 1, 14, 48, 10, 27, 13, 17, 42, 33, 42, 45, 37, 24, 45, 22, 10, 20, 40, 48, 37, 10, 16, 46, 21, 30, 44, 19, 27, 48, 9, 45, 40, 46, 43, 10, 38, 30, 15, 33, 48, 38, 10, 35, 10, 45, 2, 33, 15, 3, 34, 5, 48, 1, 3, 34, 48, 33, 30, 40, 7, 24, 48, 46, 42, 14, 4, 1, 46, 30, 35, 2, 35, 30, 42, 30, 15, 41, 37, 21, 16, 38, 43, 32, 38, 36, 37, 10, 46, 48, 19, 33, 37, 42, 4, 45, 35, 28, 1, 15, 40, 31, 38, 38, 5, 48, 29, 42, 2, 28, 15, 37, 33, 45, 38, 45, 21, 18, 45, 46, 13, 40, 10, 0, 27, 38, 46, 10, 48, 48, 17, 15, 27, 22, 19, 35, 18, 38, 33, 15, 0, 0, 43, 45, 41, 37, 48, 45, 19, 0, 15, 47, 9, 10, 49, 48, 35, 36, 48, 36, 43, 42, 19, 16, 7, 43, 16, 36, 15, 3, 21, 1, 33, 30, 1, 39, 44, 21, 28, 4, 48, 43, 48, 29, 21, 5, 10, 48, 48, 33, 48, 33, 40, 48, 25, 48, 30, 35, 30, 30, 28, 35, 42, 1, 48, 48, 19, 48, 46, 42, 28, 26, 18, 31, 10, 48, 48, 37, 44, 28, 48, 42, 46, 33, 45, 0, 0, 32, 10, 44, 24, 24, 43, 35, 11, 2, 46, 21, 38, 2, 1, 48, 21, 38, 46, 1, 34, 21, 41, 37, 20, 32, 43, 40, 49, 42, 48, 1, 48, 48, 9, 10, 33, 27, 37, 0, 8, 1, 35, 42, 13, 6, 38, 48, 26, 1, 34, 48, 48, 39, 0, 4, 18, 31, 27, 33, 4, 33, 21, 8, 15, 32, 48, 48, 13, 33, 35, 33, 10, 42, 5, 33, 33, 48, 1, 30, 3, 35, 9, 0, 48, 10, 10, 42, 48, 46, 15, 10, 33, 46, 45, 10, 0, 9, 41, 33, 10, 49, 4, 49, 48, 26, 17, 33, 1, 46, 4, 9, 1, 0, 38, 42, 46, 48, 10, 0, 0, 48, 8, 10, 0, 48, 29, 45, 27, 38, 33, 35, 38, 48, 0, 44, 48, 0, 10, 48, 38, 46, 42, 33, 48, 39, 36, 19, 45, 33, 45, 10, 33, 35, 40, 23, 33, 44, 7, 1, 10, 47, 44, 15, 15, 3, 48, 32, 39, 0, 32, 0, 35, 41, 48, 21, 47, 48, 46, 37, 46, 23, 16, 40, 48, 4, 46, 2, 36, 42, 24, 48, 15, 26, 10, 0, 46, 46, 42, 4, 30, 40, 46, 0, 10, 44, 38, 0, 46, 30, 33, 22, 48, 0, 46, 32, 12, 48, 37, 15, 33, 12, 7, 38, 45, 21, 38, 0, 48, 41, 10, 27, 48, 10, 33, 4, 12, 5, 43, 32, 49, 1, 21, 43, 30, 48, 15, 33, 19, 19, 48, 33, 1, 2, 3, 10, 32, 10, 38, 18, 28, 5, 48, 40, 48, 38, 5, 39, 6, 48, 9, 37, 17, 18, 48, 48, 38, 37, 17, 43, 11, 35, 24, 0, 8, 12, 32, 4, 32, 10, 0, 10, 33, 27, 34, 7, 0, 10, 30, 35, 0, 43, 40, 16, 16, 48, 19, 10, 44, 10, 5, 18, 14, 35, 4, 46, 10, 42, 22, 28, 16, 4, 8, 30, 33, 10, 45, 36, 45, 48, 47, 13, 18, 48, 35, 31, 16, 10, 10, 45, 5, 33, 10, 0, 43, 49, 9, 15, 11, 15, 19, 31, 2, 46, 2, 10, 43, 33, 33, 28, 35, 38, 21, 3, 42, 10, 40, 35, 43, 27, 15, 47, 36, 34, 0, 26, 21, 48, 1, 42, 15, 38, 46, 24, 42, 29, 36, 48, 33, 1, 48, 30, 35, 0, 48, 35, 0, 10, 30, 46, 4, 32, 41, 43, 12, 16, 16, 36, 10, 0, 29, 48, 40, 35, 33, 9, 27, 1, 38, 36, 30, 0, 38, 19, 3, 48, 48, 43, 0, 16, 29, 7, 21, 36, 27, 48, 48, 35, 30, 10, 41, 10, 19, 49, 0, 33, 35, 16, 48, 38, 10, 10, 48, 41, 48, 3, 41, 13, 32, 2, 2, 31, 48, 31, 38, 10, 5, 14, 27, 46, 26, 48, 28, 35, 16, 30, 36, 10, 37, 45, 35, 40, 4, 46, 49, 4, 4, 30, 29, 36, 47, 10, 48, 28, 22, 13, 33, 10, 0, 1, 28, 27, 29, 42, 45, 10, 32, 45, 10, 46, 45, 22, 45, 33, 21, 41, 10, 1, 48, 37, 19, 43, 48, 7, 32, 38, 30, 45, 42, 29, 2, 30, 27, 38, 40, 10, 43, 29, 11, 1, 22, 46, 2, 33, 0, 9, 43, 34, 10, 4, 21, 38, 46, 36, 13, 23, 22, 48, 42, 12, 31, 2, 30, 37, 15, 8, 31, 33, 33, 11, 39, 48, 48, 1, 49, 0, 29, 0, 13, 15, 28, 33, 15, 10, 48, 1, 48, 0, 48, 2, 46, 33, 48, 37, 32, 4, 21, 21, 15, 15, 19, 43, 38, 10, 10, 5, 30, 43, 38, 29, 10, 48, 48, 48, 42, 39, 15, 6, 14, 21, 10, 10, 10, 42, 28, 48, 48, 0, 35, 40, 45, 28, 10, 48, 10, 10, 38, 10, 10, 15, 7, 1, 40, 26, 10, 48, 15, 29, 4, 43, 21, 18, 31, 44, 16, 33, 20, 42, 38, 3, 4, 24, 48, 10, 46, 10, 36, 18, 28, 1, 38, 15, 10, 48, 44, 44, 0, 3, 36, 43, 40, 48, 37, 0, 38, 19, 2, 21, 1, 28, 41, 10, 30, 10, 19, 48, 45, 31, 48, 2, 33, 12, 13, 47, 18, 2, 41, 15, 33, 48, 18, 2, 36, 33, 48, 44, 15, 1, 42, 25, 10, 48, 47, 48, 11, 48, 10, 43, 40, 42, 42, 3, 48, 16, 13, 42, 48, 15, 33, 30, 43, 11, 48, 1, 48, 9, 46, 21, 21, 48, 12, 48, 26, 33, 46, 48, 10, 33, 44, 33, 38, 24, 37, 46, 8, 18, 29, 48, 44, 4, 39, 32, 45, 18, 43, 38, 48, 38, 1, 49, 15, 31, 48, 41, 37, 48, 30, 42, 32, 18, 48, 48, 38, 21, 0, 31, 11, 2, 26, 28, 35, 15, 0, 28, 1, 9, 33, 18, 44, 42, 0, 1, 42, 10, 29, 10, 10, 2, 31, 2, 7, 10, 9, 35, 32, 48, 38, 0, 27, 5, 15, 18, 48, 48, 9, 15, 38, 48, 46, 11, 48, 19, 44, 32, 47, 10, 43, 33, 43, 46, 42, 33, 0, 15, 47, 42, 21, 10, 42, 0, 48, 33, 0, 43, 48, 49, 43, 48, 40, 35, 1, 36, 23, 48, 46, 10, 10, 27, 1, 10, 15, 28, 46, 11, 36, 0, 10, 40, 47, 15, 1, 48, 38, 6, 32, 10, 0, 38, 35, 35, 0, 10, 4, 15, 39, 37, 32, 26, 1, 2, 43, 48, 30, 41, 4, 0, 2, 18, 48, 8, 40, 1, 36, 48, 45, 41, 35, 43, 33, 15, 46, 0, 38, 33, 38, 29, 29, 37, 48, 35, 44, 33, 10, 29, 19, 40, 0, 28, 21, 48, 40, 10, 33, 39, 45, 38, 33, 15, 48, 30, 48, 36, 18, 10, 10, 35, 28, 2, 43, 28, 10, 35, 21, 36, 48, 2, 48, 46, 0, 38, 32, 21, 2, 15, 46, 48, 1, 48, 29, 44, 7, 28, 26, 40, 48, 10, 33, 12, 37, 3, 37, 29, 48, 19, 37, 49, 10, 16, 41, 41, 33, 46, 21, 35, 22, 16, 5, 38, 36, 10, 48, 10, 24, 46, 48, 48, 42, 41, 19, 7, 44, 1, 38, 45, 38, 10, 29, 10, 38, 43, 42, 46, 0, 38, 27, 43, 48, 35, 35, 24, 35, 48, 10, 19, 48, 24, 10, 33, 37, 2, 24, 10, 31, 10, 47, 15, 21, 18, 30, 22, 1, 48, 48, 23, 1, 45, 47, 15, 35, 48, 24, 1, 30, 10, 48, 10, 10, 29, 35, 47, 46, 38, 10, 48, 48, 43, 0, 48, 39, 36, 4, 29, 21, 36, 48, 37, 2, 25, 42, 35, 10, 18, 30, 0, 48, 34, 3, 10, 10, 48, 46, 4, 23, 15, 8, 38, 15, 45, 4, 29, 10, 38, 45, 18, 43, 17, 7, 10, 45, 1, 41, 42, 38, 10, 36, 40, 21, 35, 48, 15, 48, 0, 46, 45, 42, 33, 33, 35, 41, 10, 32, 36, 48, 31, 36, 13, 45, 16, 46, 10, 46, 22, 40, 7, 2, 1, 0, 38, 19, 8, 32, 27, 35, 5, 10, 10, 10, 2, 2, 4, 47, 34, 42, 19, 0, 23, 9, 0, 42, 43, 33, 48, 48, 38, 2, 43, 18, 48, 0, 40, 10, 0, 36, 10, 35, 10, 48, 21, 4, 16, 13, 39, 30, 10, 10, 48, 48, 21, 44, 48, 26, 19, 32, 33, 4, 30, 10, 42, 45, 47, 48, 1, 10, 1, 29, 48, 15, 32, 19, 10, 29, 9, 19, 40, 32, 45, 18, 10, 27, 32, 40, 46, 33, 16, 10, 29, 38, 41, 30, 33, 2, 21, 48, 36, 41, 44, 42, 13, 5, 38, 46, 1, 3, 37, 46, 46, 22, 1, 35, 3, 3, 48, 10, 27, 36, 18, 47, 45, 10, 42, 33, 48, 15, 39, 0, 21, 0, 6, 0, 18, 40, 46, 48, 2, 19, 46, 48, 33, 1, 42, 12, 1, 0, 24, 48, 15, 24, 29, 10, 10, 26, 21, 16, 0, 17, 35, 23, 34, 45, 37, 18, 48, 9, 46, 43, 8, 0, 30, 33, 48, 21, 21, 48, 15, 37, 38, 38, 24, 48, 1, 11, 0, 18, 42, 36, 30, 15, 37, 43, 0, 0, 11, 48, 43, 43, 15, 45, 35, 41, 21, 40, 10, 10, 48, 42, 1, 48, 10, 48, 9, 4, 21, 7, 10, 15, 15, 38, 40, 15, 43, 40, 40, 2, 28, 35, 6, 1, 33, 46, 44, 21, 43, 46, 48, 9, 42, 10, 10, 42, 24, 33, 16, 9, 0, 36, 15, 1, 44, 0, 47, 33, 12, 48, 46, 32, 48, 1, 21, 38, 48, 15, 3, 21, 23, 38, 28, 18, 10, 10, 30, 43, 10, 15, 23, 32, 5, 10, 35, 34, 41, 41, 38, 48, 24, 48, 42, 45, 42, 10, 21, 8, 32, 42, 12, 48, 28, 13, 1, 19, 33, 40, 37, 8, 19, 18, 30, 29, 10, 43, 8, 38, 42, 39, 48, 49, 0, 38, 37, 28, 33, 44, 48, 38, 27, 46, 48, 38, 9, 8, 4, 48, 12, 16, 37, 45, 40, 2, 10, 29, 46, 1, 0, 28, 15, 35, 38, 1, 40, 8, 1, 19, 2, 33, 33, 48, 4, 6, 9, 29, 32, 1, 10, 1, 10, 2, 36, 48, 45, 38, 40, 45, 46, 0, 15, 0, 42, 27, 44, 47, 2, 10, 29, 33, 11, 11, 41, 35, 44, 45, 48, 1, 16, 31, 4, 36, 41, 0, 11, 8, 46, 40, 40, 43, 11, 33, 47, 15, 3, 6, 24, 10, 35, 23, 37, 16, 30, 16, 37, 35, 48, 33, 38, 15, 33, 38, 3, 46, 37, 35, 4, 30, 1, 33, 48, 29, 10, 4, 3, 2, 48, 29, 17, 30, 10, 12, 38, 41, 0, 3, 41, 2, 0, 35, 33, 45, 24, 48, 10, 26, 1, 1, 45, 21, 38, 38, 23, 37, 30, 49, 38, 9, 10, 35, 46, 35, 44, 15, 33, 48, 10, 10, 19, 37, 10, 48, 40, 35, 30, 9, 36, 1, 1, 41, 47, 38, 36, 17, 15, 46, 36, 15, 10, 10, 33, 39, 46, 21, 15, 38, 19, 1, 48, 2, 10, 19, 1, 0, 36, 12, 38, 26, 35, 28, 10, 15, 2, 10, 32, 29, 18, 2, 0, 43, 10, 28, 13, 2, 36, 48, 24, 32, 15, 46, 32, 24, 33, 15, 7, 32, 48, 13, 45, 43, 13, 43, 43, 48, 48, 15, 33, 1, 37, 46, 42, 10, 30, 41, 17, 5, 10, 10, 48, 35, 21, 19, 42, 5, 10, 33, 35, 40, 0, 21, 36, 17, 41, 33, 0, 35, 49, 44, 1, 32, 33, 40, 24, 0, 48, 33, 1, 0, 21, 48, 0, 24, 36, 42, 37, 21, 28, 46, 1, 38, 46, 27, 15, 9, 1, 1, 48, 2, 44, 35, 2, 30, 32, 7, 3, 38, 34, 11, 10, 10, 36, 45, 46, 48, 39, 10, 49, 44, 47, 24, 38, 40, 45, 48, 15, 10, 0, 10, 48, 2, 44, 0, 10, 42, 17, 30, 10, 29, 21, 15, 48, 32, 42, 15, 1, 2, 26, 45, 48, 15, 10, 48, 18, 36, 43, 34, 42, 10, 45, 43, 1, 15, 40, 37, 36, 32, 19, 29, 37, 41, 1, 0, 48, 37, 37, 0, 1, 31, 9, 35, 3, 33, 21, 48, 35, 21, 48, 29, 10, 0, 21, 40, 48, 0, 23, 31, 0, 48, 42, 38, 30, 3, 33, 15, 42, 0, 48, 48, 31, 2, 20, 38, 49, 10, 35, 15, 26, 40, 1, 28, 33, 38, 21, 46, 35, 41, 48, 48, 38, 46, 0, 38, 10, 0, 16, 40, 47, 40, 10, 48, 35, 15, 11, 38, 9, 30, 42, 15, 13, 2, 0, 48, 42, 33, 29, 4, 28, 0, 38, 10, 42, 22, 35, 0, 33, 21, 3, 44, 33, 3, 33, 0, 30, 49, 18, 48, 1, 5, 15, 10, 46, 4, 43, 19, 43, 27, 36, 35, 34, 13, 35, 10, 43, 34, 48, 15, 0, 37, 45, 0, 9, 0, 15, 48, 15, 10, 0, 33, 0, 1, 37, 48, 48, 30, 13, 42, 24, 42, 45, 49, 33, 1, 1, 10, 3, 10, 4, 38, 35, 10, 15, 47, 47, 48, 28, 42, 35, 42, 48, 30, 40, 2, 36, 40, 17, 48, 40, 10, 15, 10, 43, 48, 1, 0, 46, 14, 9, 5, 19, 28, 37, 42, 36, 10, 10, 47, 34, 0, 0, 19, 23, 38, 8, 35, 10, 38, 7, 2, 6, 10, 39, 48, 4, 10, 42, 11, 0, 2, 33, 0, 21, 4, 0, 10, 2, 45, 42, 0, 11, 15, 39, 0, 1, 10, 46, 18, 1, 30, 30, 33, 38, 45, 48, 10, 0, 15, 0, 20, 40, 40, 33, 7, 35, 22, 3, 29, 35, 33, 37, 48, 0, 10, 12, 48, 42, 44, 48, 4, 46, 13, 46, 18, 38, 46, 41, 41, 41, 38, 26, 48, 29, 15, 48, 28, 48, 48, 24, 22, 19, 10, 8, 2, 18, 5, 45, 15, 38, 29, 31, 36, 35, 0, 46, 47, 8, 48, 0, 33, 15, 35, 10, 0, 12, 10, 24, 47, 0, 32, 46, 33, 36, 45, 7, 46, 39, 48, 15, 21, 10, 10, 28, 43, 8, 48, 37, 16, 10, 10, 40, 48, 15, 19, 36, 10, 19, 33, 48, 33, 1, 49, 48, 2, 24, 42, 47, 46, 37, 10, 10, 22, 48, 42, 33, 30, 49, 36, 38, 33, 16, 24, 16, 33, 10, 10, 43, 40, 3, 45, 21, 21, 10, 15, 30, 48, 1, 9, 48, 42, 43, 30, 33, 40, 46, 15, 48, 21, 20, 46, 0, 48, 10, 26, 37, 38, 42, 38, 43, 32, 35, 35, 46, 45, 10, 33, 10, 42, 42, 49, 9, 38, 48, 41, 30, 10, 2, 42, 4, 0, 48, 4, 48, 33, 48, 0, 38, 30, 34, 33, 46, 33, 9, 48, 29, 48, 10, 5, 40, 48, 48, 10, 13, 48, 48, 5, 30, 10, 9, 2, 21, 31, 22, 2, 47, 42, 42, 38, 24, 38, 3, 30, 21, 33, 10, 48, 28, 30, 15, 48, 1, 15, 10, 13, 48, 21, 41, 36, 16, 26, 41, 15, 42, 24, 13, 20, 47, 1, 10, 48, 15, 1, 48, 28, 1, 48, 16, 43, 10, 44, 16, 38, 6, 9, 25, 43, 35, 41, 46, 18, 37, 0, 46, 16, 42, 7, 1, 1, 21, 19, 38, 35, 19, 4, 48, 4, 48, 33, 12, 0, 10, 10, 10, 15, 15, 45, 37, 16, 33, 38, 0, 32, 33, 48, 48, 10, 48, 35, 16, 15, 4, 15, 0, 3, 35, 4, 0, 36, 31, 37, 0, 15, 29, 2, 13, 45, 20, 48, 35, 33, 30, 1, 48, 16, 33, 42, 45, 10, 33, 0, 10, 46, 46, 36, 42, 10, 39, 2, 15, 10, 28, 0, 9, 0, 33, 48, 44, 48, 35, 16, 48, 1, 24, 43, 28, 5, 37, 28, 1, 14, 45, 1, 29, 44, 48, 21, 48, 16, 25, 7, 18, 38, 16, 37, 2, 10, 0, 36, 38, 0, 8, 1, 2, 10, 15, 48, 21, 36, 37, 7, 33, 43, 42, 21, 1, 1, 4, 46, 24, 43, 19, 46, 2, 48, 21, 47, 43, 48, 3, 10, 10, 3, 45, 42, 13, 42, 10, 45, 5, 36, 33, 26, 43, 26, 10, 30, 1, 15, 5, 48, 10, 48, 12, 0, 33, 38, 45, 28, 37, 48, 3, 10, 46, 33, 15, 33, 29, 8, 21, 19, 44, 24, 9, 2, 0, 3, 32, 17, 43, 2, 0, 49, 46, 40, 33, 35, 43, 8, 33, 48, 19, 27, 29, 48, 46, 1, 1, 10, 33, 29, 38, 30, 30, 48, 33, 16, 7, 5, 15, 1, 15, 48, 4, 16, 49, 47, 2, 18, 24, 30, 10, 48, 36, 48, 1, 10, 33, 0, 0, 38, 8, 0, 6, 38, 36, 23, 38, 10, 35, 48, 48, 22, 46, 10, 45, 10, 7, 47, 10, 40, 4, 1, 7, 13, 30, 48, 48, 10, 39, 1, 42, 33, 42, 10, 29, 32, 10, 9, 10, 18, 41, 21, 28, 3, 17, 3, 2, 3, 10, 10, 42, 42, 48, 33, 48, 15, 2, 36, 40, 38, 12, 29, 46, 10, 4, 46, 9, 9, 28, 19, 3, 42, 32, 25, 28, 28, 42, 33, 39, 48, 32, 0, 48, 10, 48, 29, 1, 2, 2, 15, 15, 2, 0, 10, 36, 48, 38, 16, 38, 10, 33, 0, 4, 37, 0, 29, 48, 1, 40, 33, 38, 37, 48, 31, 11, 42, 46, 48, 34, 48, 0, 0, 45, 0, 48, 15, 0, 21, 45, 3, 6, 21, 38, 3, 30, 21, 46, 21, 48, 49, 16, 10, 47, 48, 10, 29, 46, 10, 8, 48, 48, 10, 1, 39, 35, 35, 48, 4, 18, 18, 30, 29, 48, 8, 17, 40, 47, 33, 40, 45, 38, 31, 2, 5, 48, 16, 18, 46, 10, 42, 39, 38, 21, 15, 16, 10, 4, 17, 10, 40, 8, 45, 29, 10, 1, 15, 45, 27, 37, 32, 4, 37, 40, 48, 22, 48, 36, 30, 18, 15, 12, 42, 19, 5, 15, 8, 48, 35, 30, 27, 46, 18, 3, 2, 30, 39, 36, 28, 42, 48, 2, 39, 15, 3, 0, 2, 10, 41, 33, 0, 10, 2, 33, 48, 36, 23, 48, 19, 38, 1, 11, 35, 42, 37, 15, 1, 33, 42, 38, 33, 10, 4, 36, 15, 1, 30, 48, 33, 10, 48, 36, 10, 45, 15, 47, 33, 46, 48, 38, 41, 13, 35, 7, 19, 48, 10, 42, 15, 29, 2, 24, 43, 9, 1, 22, 48, 27, 2, 48, 3, 0, 46, 35, 49, 45, 43, 35, 33, 26, 35, 1, 0, 15, 1, 49, 33, 30, 38, 48, 48, 48, 48, 40, 48, 38, 33, 46, 46, 30, 2, 48, 41, 18, 29, 48, 42, 36, 2, 28, 1, 10, 17, 44, 38, 48, 42, 16, 9, 13, 48, 10, 34, 36, 2, 32, 27, 16, 38, 1, 47, 2, 43, 38, 1, 22, 38, 44, 42, 36, 44, 35, 45, 18, 10, 0, 48, 18, 49, 44, 27, 11, 46, 48, 33, 12, 15, 4, 45, 30, 48, 44, 37, 0, 36, 48, 43, 42, 42, 10, 9, 8, 48, 48, 15, 33, 46, 1, 38, 8, 38, 27, 18, 1, 4, 48, 17, 42, 30, 46, 7, 48, 4, 25, 3, 2, 0, 29, 32, 1, 39, 4, 4, 42, 15, 4, 47, 4, 0, 33, 40, 35, 45, 26, 0, 46, 48, 38, 33, 2, 1, 35, 46, 45, 48, 0, 9, 18, 33, 32, 43, 12, 34, 11, 48, 42, 19, 21, 15, 46, 37, 10, 39, 10, 35, 48, 45, 48, 48, 48, 27, 43, 0, 15, 36, 10, 48, 48, 20, 43, 8, 36, 16, 14, 0, 2, 43, 21, 0, 36, 35, 43, 17, 4, 10, 29, 10, 48, 48, 46, 36, 41, 18, 4, 3, 15, 10, 0, 38, 10, 48, 6, 29, 33, 10, 15, 35, 47, 48, 30, 36, 0, 15, 10, 10, 46, 48, 38, 48, 7, 33, 43, 21, 18, 10, 40, 0, 48, 33, 11, 48, 8, 11, 11, 44, 16, 3, 46, 33, 35, 5, 10, 28, 10, 35, 45, 48, 38, 45, 45, 5, 30, 22, 26, 1, 45, 48, 48, 2, 10, 0, 31, 45, 48, 21, 10, 5, 45, 10, 44, 33, 48, 37, 33, 48, 11, 6, 13, 33, 9, 35, 9, 44, 42, 48, 27, 46, 2, 3, 10, 33, 28, 3, 0, 9, 0, 35, 42, 2, 37, 20, 48, 1, 7, 44, 33, 44, 42, 44, 43, 15, 44, 0, 32, 10, 33, 10, 38, 7, 41, 19, 40, 37, 0, 42, 23, 28, 48, 4, 11, 27, 29, 35, 36, 9, 48, 46, 37, 10, 33, 20, 43, 29, 15, 10, 48, 48, 0, 32, 9, 43, 1, 30, 48, 33, 42, 47, 24, 6, 32, 43, 9, 48, 10, 13, 46, 2, 16, 3, 48, 41, 30, 37, 9, 10, 48, 10, 38, 19, 33, 1, 32, 10, 38, 38, 43, 48, 47, 42, 1, 9, 3, 9, 5, 38, 5, 16, 33, 48, 43, 32, 0, 10, 48, 48, 35, 48, 48, 38, 46, 32, 42, 30, 0, 24, 49, 2, 39, 48, 48, 1, 43, 10, 45, 48, 15, 35, 36, 15, 46, 1, 29, 18, 0, 48, 29, 0, 35, 4, 24, 21, 10, 48, 28, 48, 10, 42, 0, 0, 32, 48, 10, 33, 12, 10, 30, 32, 19, 36, 38, 1, 0, 33, 13, 37, 48, 47, 38, 10, 8, 38, 10, 33, 46, 44, 18, 41, 48, 27, 8, 10, 10, 48, 33, 10, 41, 46, 28, 6, 28, 46, 7, 44, 10, 8, 1, 28, 21, 48, 40, 0, 38, 43, 19, 32, 16, 35, 46, 38, 0, 1, 0, 23, 32, 48, 42, 7, 0, 35, 40, 48, 33, 10, 18, 0, 48, 48, 33, 14, 46, 10, 22, 29, 37, 10, 21, 46, 38, 21, 46, 8, 10, 19, 49, 10, 10, 46, 12, 46, 42, 48, 35, 33, 1, 0, 46, 46, 0, 17, 42, 6, 15, 0, 4, 10, 37, 16, 48, 2, 38, 48, 15, 12, 0, 48, 1, 10, 33, 5, 3, 42, 16, 32, 46, 9, 33, 4, 40, 48, 10, 48, 27, 4, 8, 45, 26, 10, 42, 40, 27, 28, 10, 12, 35, 23, 48, 27, 33, 15, 38, 35, 18, 10, 21, 33, 32, 38, 15, 33, 46, 38, 30, 37, 48, 48, 0, 16, 42, 38, 36, 41, 10, 5, 39, 10, 2, 42, 32, 6, 33, 21, 40, 39, 38, 32, 48, 43, 33, 4, 38, 15, 0, 19, 6, 15, 2, 34, 21, 10, 1, 4, 11, 27, 33, 48, 22, 15, 48, 4, 40, 1, 36, 32, 43, 38, 10, 40, 1, 33, 48, 27, 19, 27, 22, 13, 43, 1, 42, 32, 30, 18, 4, 48, 4, 10, 12, 1, 12, 48, 37, 48, 1, 15, 19, 38, 33, 10, 40, 46, 0, 18, 10, 4, 42, 1, 18, 37, 36, 11, 0, 48, 2, 46, 43, 48, 25, 20, 15, 10, 32, 46, 6, 5, 40, 39, 18, 27, 44, 42, 46, 9, 38, 6, 33, 44, 10, 42, 1, 35, 26, 42, 3, 48, 48, 2, 22, 37, 41, 36, 48, 33, 43, 4, 10, 9, 10, 33, 4, 15, 45, 21, 15, 38, 10, 10, 30, 41, 38, 40, 44, 38, 29, 24, 42, 19, 1, 9, 43, 15, 33, 41, 8, 40, 46, 16, 35, 1, 22, 0, 42, 30, 3, 26, 11, 36, 1, 44, 37, 16, 38, 48, 10, 38, 2, 4, 46, 3, 33, 35, 37, 41, 40, 48, 10, 24, 10, 15, 4, 48, 32, 0, 37, 47, 33, 32, 2, 48, 48, 18, 2, 4, 0, 27, 37, 0, 48, 21, 48, 32, 22, 38, 19, 16, 48, 36, 27, 32, 29, 21, 42, 0, 46, 33, 47, 40, 10, 36, 1, 37, 35, 4, 4, 35, 1, 35, 44, 1, 11, 41, 0, 2, 30, 35, 23, 44, 36, 48, 35, 1, 42, 22, 40, 35, 3, 19, 44, 48, 33, 37, 0, 0, 15, 15, 48, 18, 49, 1, 41, 21, 30, 33, 18, 42, 40, 44, 46, 43, 18, 40, 37, 35, 10, 2, 10, 33, 46, 47, 35, 38, 1, 13, 48, 27, 5, 34, 10, 33, 48, 35, 15, 48, 33, 43, 19, 43, 16, 42, 48, 18, 10, 1, 38, 10, 10, 10, 32, 38, 10, 33, 41, 30, 13, 2, 48, 20, 10, 37, 0, 36, 10, 30, 1, 11, 42, 36, 45, 35, 12, 0, 44, 48, 48, 0, 0, 37, 4, 9, 15, 47, 45, 32, 19, 15, 44, 38, 46, 40, 18, 38, 43, 49, 46, 10, 40, 36, 15, 38, 33, 19, 10, 19, 2, 36, 48, 16, 10, 38, 41, 38, 1, 48, 43, 4, 15, 38, 10, 48, 38, 46, 41, 6, 36, 27, 10, 30, 33, 2, 48, 35, 13, 24, 36, 20, 7, 26, 42, 10, 34, 1, 48, 35, 38, 10, 15, 35, 35, 48, 21, 39, 29, 33, 16, 40, 4, 6, 8, 38, 15, 46, 38, 38, 44, 48, 1, 46, 42, 22, 10, 18, 38, 21, 2, 8, 0, 30, 43, 47, 48, 49, 15, 15, 48, 48, 32, 17, 3, 44, 33, 48, 10, 1, 47, 16, 42, 48, 35, 15, 10, 32, 36, 45, 40, 9, 48, 30, 10, 10, 48, 24, 30, 10, 39, 10, 30, 15, 21, 42, 2, 0, 38, 1, 48, 21, 27, 39, 37, 35, 3, 16, 43, 33, 44, 5, 3, 48, 1, 48, 10, 39, 36, 48, 48, 37, 15, 43, 21, 45, 44, 42, 0, 10, 10, 30, 28, 16, 44, 18, 45, 43, 42, 1, 38, 48, 9, 33, 15, 33, 13, 12, 5, 15, 49, 12, 10, 18, 48, 48, 2, 36, 45, 15, 24, 0, 26, 46, 45, 44, 18, 38, 48, 8, 36, 40, 36, 40, 0, 29, 36, 33, 21, 21, 48, 12, 15, 1, 16, 32, 38, 42, 38, 48, 48, 48, 48, 9, 39, 2, 25, 48, 27, 48, 37, 49, 27, 35, 29, 16, 16, 48, 46, 48, 32, 19, 38, 46, 43, 48, 37, 10, 48, 1, 23, 38, 10, 32, 12, 48, 5, 42, 15, 35, 1, 46, 42, 2, 0, 29, 48, 9, 48, 43, 1, 9, 35, 8, 10, 35, 21, 48, 43, 18, 45, 4, 48, 21, 15, 48, 9, 29, 46, 32, 13, 1, 0, 10, 48, 34, 10, 7, 21, 33, 10, 26, 33, 9, 42, 33, 38, 38, 48, 3, 19, 37, 1, 49, 41, 33, 0, 30, 48, 15, 46, 33, 46, 48, 10, 40, 35, 2, 10, 10, 12, 46, 2, 45, 46, 37, 35, 48, 24, 10, 33, 45, 31, 39, 40, 42, 48, 15, 13, 10, 38, 48, 44, 10, 42, 36, 16, 20, 33, 48, 0, 22, 3, 0, 37, 10, 35, 33, 2, 19, 42, 10, 10, 16, 48, 33, 0, 35, 41, 28, 33, 4, 10, 48, 7, 42, 1, 41, 10, 38, 0, 49, 24, 1, 9, 9, 49, 15, 0, 35, 44, 48, 37, 12, 39, 19, 33, 13, 10, 16, 31, 2, 48, 0, 5, 16, 1, 7, 10, 16, 33, 2, 21, 15, 9, 43, 45, 0, 33, 4, 43, 27, 3, 19, 44, 43, 15, 40, 0, 45, 18, 38, 0, 44, 49, 28, 30, 35, 23, 10, 45, 37, 38, 1, 10, 10, 1, 33, 46, 0, 33, 0, 35, 15, 27, 38, 35, 46, 48, 0, 43, 1, 20, 48, 43, 43, 9, 15, 10, 48, 42, 10, 10, 41, 23, 10, 10, 46, 42, 48, 9, 36, 10, 40, 45, 10, 30, 37, 46, 30, 24, 30, 41, 44, 33, 10, 0, 29, 2, 49, 40, 43, 33, 45, 0, 4, 10, 15, 45, 9, 41, 38, 35, 46, 48, 46, 48, 10, 38, 38, 38, 32, 48, 32, 2, 1, 48, 24, 1, 48, 10, 10, 48, 37, 48, 48, 35, 1, 1, 46, 43, 9, 38, 10, 28, 46, 23, 13, 48, 39, 29, 49, 27, 46, 38, 35, 28, 10, 46, 30, 46, 9, 18, 15, 12, 0, 38, 48, 48, 48, 42, 12, 48, 38, 45, 15, 0, 19, 18, 33, 31, 12, 48, 15, 4, 43, 48, 15, 49, 17, 46, 33, 27, 42, 10, 30, 36, 32, 36, 9, 46, 10, 8, 15, 24, 10, 45, 15, 38, 4, 48, 48, 4, 22, 9, 30, 1, 11, 16, 10, 2, 1, 38, 29, 21, 42, 29, 2, 41, 10, 48, 10, 30, 38, 30, 38, 38, 29, 38, 18, 3, 10, 39, 33, 15, 48, 49, 1, 16, 44, 35, 18, 30, 4, 18, 23, 17, 37, 49, 19, 0, 13, 21, 15, 15, 0, 34, 36, 42, 19, 0, 36, 5, 48, 48, 21, 22, 48, 43, 17, 36, 40, 48, 35, 48, 10, 35, 30, 48, 48, 10, 45, 29, 18, 30, 1, 48, 35, 21, 33, 48, 10, 10, 33, 21, 35, 35, 33, 11, 15, 19, 1, 48, 11, 42, 15, 38, 44, 32, 0, 30, 43, 47, 49, 38, 39, 0, 10, 9, 4, 15, 15, 16, 1, 15, 43, 42, 1, 1, 7, 29, 36, 47, 24, 46, 22, 11, 33, 7, 1, 10, 19, 1, 35, 10, 1, 48, 46, 29, 41, 33, 10, 3, 28, 47, 7, 1, 42, 43, 40, 22, 42, 38, 30, 30, 17, 1, 26, 30, 35, 6, 48, 38, 48, 37, 40, 47, 30, 38, 13, 38, 33, 24, 43, 48, 43, 31, 32, 42, 18, 9, 46, 33, 23, 38, 48, 36, 40, 8, 1, 2, 19, 3, 40, 35, 43, 38, 47, 48, 43, 32, 46, 48, 38, 48, 21, 29, 18, 43, 2, 1, 22, 44, 1, 27, 17, 41, 49, 33, 37, 2, 26, 32, 32, 15, 10, 32, 11, 48, 38, 35, 15, 42, 16, 29, 10, 19, 48, 46, 43, 35, 48, 36, 11, 2, 0, 2, 4, 38, 46, 19, 0, 15, 13, 15, 15, 44, 40, 18, 45, 48, 10, 26, 46, 24, 46, 10, 43, 10, 46, 38, 35, 48, 34, 2, 38, 2, 33, 38, 38, 15, 43, 2, 43, 1, 10, 30, 48, 8, 0, 48, 16, 36, 44, 45, 9, 41, 4, 42, 0, 7, 31, 48, 42, 15, 38, 35, 48, 33, 42, 48, 28, 42, 48, 4, 48, 19, 48, 4, 40, 28, 26, 4, 4, 15, 16, 10, 24, 24, 38, 48, 43, 10, 48, 4, 21, 2, 9, 21, 21, 48, 43, 37, 30, 2, 4, 45, 45, 48, 18, 46, 38, 19, 21, 12, 1, 1, 27, 32, 10, 48, 48, 9, 10, 37, 26, 29, 22, 2, 42, 10, 11, 15, 43, 10, 48, 10, 33, 45, 46, 0, 29, 40, 32, 27, 48, 0, 48, 47, 15, 2, 21, 48, 3, 10, 10, 29, 15, 1, 38, 30, 46, 15, 10, 9, 48, 46, 21, 46, 48, 10, 1, 0, 18, 43, 40, 29, 33, 48, 48, 33, 37, 2, 10, 9, 23, 35, 45, 40, 32, 41, 15, 48, 10, 10, 10, 0, 10, 48, 30, 10, 18, 1, 46, 10, 35, 33, 35, 46, 15, 10, 16, 5, 7, 10, 45, 1, 29, 43, 48, 0, 38, 33, 48, 7, 18, 48, 28, 46, 43, 40, 1, 45, 40, 33, 18, 27, 27, 45, 10, 4, 4, 35, 10, 44, 36, 43, 35, 15, 32, 3, 13, 38, 10, 35, 22, 37, 48, 35, 42, 29, 35, 9, 10, 11, 10, 7, 35, 4, 29, 9, 31, 38, 18, 43, 45, 12, 32, 1, 46, 38, 44, 1, 4, 30, 15, 27, 0, 26, 16, 1, 30, 42, 37, 48, 33, 10, 1, 2, 48, 29, 48, 28, 0, 42, 46, 40, 46, 29, 45, 37, 6, 10, 48, 36, 43, 10, 32, 35, 42, 17, 32, 48, 47, 46, 39, 49, 48, 46, 44, 5, 48, 2, 10, 25, 45, 21, 32, 24, 41, 46, 10, 32, 10, 41, 28, 16, 15, 15, 7, 6, 1, 6, 31, 41, 48, 21, 1, 36, 13, 38, 15, 29, 10, 32, 47, 10, 15, 48, 35, 0, 48, 48, 1, 42, 3, 30, 17, 37, 1, 0, 13, 28, 33, 7, 1, 33, 0, 48, 48, 28, 30, 42, 42, 49, 47, 48, 36, 37, 46, 10, 0, 48, 6, 33, 40, 43, 10, 44, 24, 36, 38, 29, 30, 45, 49, 46, 12, 10, 33, 35, 10, 0, 48, 11, 43, 35, 44, 26, 0, 47, 48, 38, 37, 33, 43, 1, 38, 13, 10, 17, 10, 33, 5, 10, 44, 48, 11, 0, 2, 1, 48, 48, 38, 6, 19, 9, 10, 1, 10, 15, 30, 49, 33, 13, 33, 43, 38, 20, 44, 48, 1, 42, 30, 28, 2, 46, 35, 19, 48, 30, 0, 30, 11, 48, 35, 5, 18, 44, 42, 38, 1, 48, 10, 48, 25, 43, 33, 3, 18, 12, 0, 33, 44, 33, 46, 33, 10, 24, 7, 48, 49, 45, 48, 33, 10, 32, 48, 48, 41, 49, 7, 6, 4, 38, 10, 2, 2, 23, 48, 1, 48, 33, 38, 6, 10, 48, 10, 36, 19, 0, 35, 2, 48, 48, 48, 16, 32, 4, 1, 37, 36, 48, 0, 41, 0, 5, 42, 10, 38, 35, 48, 49, 13, 47, 15, 48, 48, 2, 48, 30, 0, 12, 36, 22, 48, 47, 18, 9, 35, 38, 1, 1, 9, 48, 17, 48, 1, 45, 24, 16, 10, 38, 48, 48, 33, 4, 35, 47, 35, 35, 10, 18, 33, 33, 48, 33, 15, 11, 10, 15, 1, 38, 42, 2, 4, 48, 42, 10, 38, 21, 4, 13, 16, 0, 16, 42, 1, 38, 9, 15, 49, 33, 2, 41, 46, 1, 33, 1, 35, 33, 33, 46, 38, 21, 35, 43, 48, 35, 3, 2, 18, 15, 24, 1, 7, 10, 46, 0, 16, 35, 3, 48, 31, 1, 35, 18, 15, 4, 34, 19, 9, 10, 38, 21, 13, 15, 49, 26, 37, 27, 48, 29, 47, 10, 36, 35, 36, 10, 10, 32, 7, 11, 11, 35, 8, 38, 33, 40, 47, 48, 15, 26, 0, 48, 20, 47, 10, 3, 0, 21, 33, 30, 33, 10, 15, 21, 48, 1, 35, 0, 42, 15, 43, 36, 43, 4, 7, 18, 18, 33, 40, 35, 19, 19, 48, 45, 13, 30, 39, 19, 1, 45, 45, 10, 33, 41, 28, 38, 38, 24, 15, 40, 15, 47, 17, 22, 48, 23, 33, 37, 42, 0, 8, 42, 35, 48, 10, 15, 42, 49, 38, 46, 48, 0, 45, 16, 48, 5, 9, 33, 45, 15, 18, 15, 10, 46, 39, 15, 1, 48, 17, 37, 10, 48, 18, 15, 43, 38, 33, 0, 12, 1, 30, 32, 43, 10, 10, 44, 16, 45, 34, 30, 38, 10, 36, 13, 0, 43, 43, 42, 0, 32, 2, 41, 1, 18, 21, 0, 47, 40, 17, 35, 24, 48, 48, 48, 46, 48, 48, 32, 0, 0, 2, 31, 36, 35, 10, 9, 1, 1, 48, 2, 18, 10, 46, 30, 42, 21, 49, 1, 35, 13, 40, 6, 46, 19, 27, 38, 0, 45, 37, 48, 46, 16, 46, 12, 36, 47, 49, 21, 32, 21, 32, 42, 33, 42, 18, 47, 44, 48, 43, 29, 11, 49, 1, 39, 48, 9, 38, 38, 18, 17, 43, 37, 11, 17, 31, 35, 1, 13, 34, 15, 1, 48, 4, 2, 1, 32, 16, 4, 19, 22, 7, 10, 43, 0, 48, 1, 15, 19, 33, 19, 35, 38, 11, 16, 40, 12, 41, 36, 37, 29, 46, 48, 6, 28, 42, 38, 48, 35, 10, 33, 18, 38, 30, 16, 13, 38, 42, 35, 34, 48, 47, 10, 42, 46, 30, 28, 48, 31, 48, 0, 10, 21, 0, 10, 48, 48, 44, 1, 17, 30, 48, 15, 26, 10, 1, 44, 15, 48, 18, 42, 40, 15, 0, 48, 42, 1, 13, 10, 40, 38, 10, 38, 36, 2, 48, 10, 48, 45, 43, 18, 10, 3, 24, 42, 6, 1, 47, 10, 19, 49, 46, 32, 48, 9, 27, 29, 43, 46, 28, 37, 43, 5, 43, 29, 2, 0, 49, 48, 37, 33, 30, 3, 2, 41, 48, 36, 48, 48, 37, 37, 48, 38, 0, 32, 17, 16, 21, 27, 32, 2, 1, 48, 10, 6, 42, 11, 32, 33, 35, 48, 1, 30, 48, 32, 48, 9, 42, 35, 10, 38, 45, 10, 1, 49, 0, 46, 46, 3, 49, 4, 38, 32, 26, 15, 8, 36, 1, 19, 9, 40, 38, 44, 10, 3, 31, 10, 38, 33, 30, 43, 10, 48, 12, 19, 1, 12, 35, 8, 6, 16, 10, 18, 13, 46, 29, 42, 19, 39, 48, 27, 16, 9, 48, 48, 33, 6, 0, 16, 48, 48, 9, 2, 48, 10, 10, 18, 27, 48, 48, 10, 42, 21, 45, 2, 10, 36, 10, 8, 44, 35, 42, 33, 48, 4, 46, 42, 48, 21, 1, 6, 33, 10, 10, 11, 46, 21, 11, 10, 27, 36, 10, 39, 19, 42, 33, 33, 48, 1, 48, 13, 10, 48, 18, 41, 0, 46, 10, 10, 22, 2, 4, 9, 28, 42, 15, 10, 3, 48, 8, 21, 2, 7, 48, 0, 2, 36, 31, 38, 13, 15, 4, 48, 37, 1, 48, 49, 8, 10, 42, 15, 48, 32, 6, 46, 15, 8, 49, 18, 26, 15, 1, 48, 48, 37, 0, 10, 27, 15, 2, 35, 31, 40, 0, 15, 32, 1, 42, 35, 2, 48, 28, 1, 22, 48, 36, 29, 24, 43, 33, 42, 15, 33, 9, 21, 10, 10, 43, 36, 42, 43, 7, 1, 2, 36, 49, 33, 38, 13, 0, 42, 10, 43, 15, 10, 10, 15, 46, 10, 1, 38, 10, 48, 38, 4, 48, 10, 8, 40, 10, 48, 33, 10, 1, 4, 29, 21, 1, 30, 42, 17, 18, 43, 48, 37, 48, 7, 35, 37, 35, 46, 46, 38, 48, 45, 41, 37, 48, 0, 18, 23, 19, 21, 19, 48, 48, 19, 37, 7, 18, 40, 10, 21, 46, 39, 19, 22, 46, 22, 38, 21, 28, 41, 38, 1, 15, 48, 33, 48, 30, 1, 43, 33, 30, 48, 36, 16, 3, 10, 16, 15, 10, 42, 37, 40, 44, 10, 2, 10, 10, 11, 10, 10, 37, 26, 30, 29, 36, 19, 24, 42, 4, 43, 21, 43, 18, 3, 24, 38, 12, 30, 24, 38, 10, 48, 10, 48, 0, 48, 30, 33, 49, 10, 35, 10, 31, 30, 9, 48, 9, 45, 35, 21, 41, 8, 29, 48, 28, 48, 29, 39, 48, 15, 0, 21, 48, 27, 19, 41, 30, 0, 12, 38, 36, 29, 10, 15, 10, 45, 10, 33, 2, 30, 42, 16, 21, 0, 48, 10, 2, 4, 30, 18, 48, 1, 1, 22, 21, 47, 15, 24, 21, 22, 32, 48, 10, 0, 2, 21, 32, 3, 22, 7, 48, 40, 42, 42, 33, 47, 10, 45, 29, 9, 15, 10, 0, 43, 10, 5, 10, 33, 49, 34, 16, 34, 40, 47, 8, 18, 31, 19, 44, 41, 42, 37, 33, 29, 8, 37, 10, 19, 33, 46, 32, 33, 37, 39, 9, 5, 6, 46, 48, 9, 15, 19, 38, 32, 42, 10, 48, 36, 10, 33, 42, 13, 13, 15, 48, 46, 37, 0, 16, 48, 11, 15, 35, 24, 1, 44, 38, 33, 21, 46, 42, 42, 2, 24, 6, 19, 6, 48, 26, 15, 48, 48, 48, 37, 15, 21, 5, 42, 1, 22, 33, 0, 35, 0, 45, 36, 4, 10, 18, 32, 40, 16, 46, 17, 2, 1, 15, 46, 21, 37, 41, 46, 0, 10, 49, 32, 15, 4, 41, 27, 38, 10, 15, 38, 2, 21, 38, 10, 8, 0, 33, 29, 38, 4, 30, 48, 0, 1, 10, 35, 48, 48, 9, 6, 4, 10, 48, 45, 10, 2, 10, 8, 48, 38, 10, 49, 46, 44, 37, 13, 48, 4, 35, 46, 30, 33, 44, 15, 23, 0, 42, 48, 37, 21, 10, 36, 10, 0, 38, 29, 48, 3, 29, 15, 3, 31, 24, 44, 26, 13, 36, 35, 10, 8, 10, 0, 15, 18, 15, 15, 38, 30, 18, 19, 40, 38, 10, 38, 36, 20, 13, 1, 41, 0, 32, 36, 5, 45, 15, 16, 10, 23, 48, 7, 38, 21, 46, 4, 38, 26, 0, 30, 0, 4, 47, 1, 10, 15, 21, 38, 46, 17, 0, 34, 47, 1, 1, 9, 10, 10, 7, 10, 1, 35, 42, 28, 23, 30, 10, 38, 35, 46, 21, 48, 48, 1, 33, 44, 42, 48, 37, 37, 29, 19, 49, 17, 46, 10, 35, 38, 35, 19, 46, 10, 41, 10, 0, 16, 38, 9, 41, 43, 24, 1, 46, 48, 42, 7, 48, 38, 4, 37, 19, 6, 48, 1, 10, 0, 48, 0, 46, 9, 27, 41, 10, 12, 37, 49, 42, 21, 35, 17, 30, 30, 10, 42, 35, 1, 29, 38, 18, 33, 35, 22, 36, 48, 48, 49, 30, 43, 48, 4, 3, 33, 21, 48, 8, 48, 33, 42, 42, 48, 0, 15, 27, 8, 3, 48, 43, 15, 10, 15, 4, 30, 3, 21, 32, 47, 0, 1, 2, 0, 19, 33, 45, 28, 48, 10, 35, 46, 16, 17, 33, 48, 40, 1, 12, 0, 29, 10, 48, 38, 10, 42, 32, 3, 2, 46, 36, 14, 13, 35, 10, 38, 10, 40, 32, 18, 48, 48, 36, 29, 44, 33, 0, 36, 14, 48, 38, 0, 48, 48, 40, 44, 2, 15, 1, 6, 43, 46, 1, 35, 27, 18, 42, 46, 48, 6, 48, 33, 1, 44, 0, 10, 48, 46, 30, 36, 38, 49, 0, 10, 13, 1, 48, 21, 10, 1, 40, 44, 20, 38, 38, 45, 19, 19, 2, 41, 17, 46, 40, 21, 24, 37, 31, 18, 5, 11, 22, 4, 0, 46, 17, 23, 27, 48, 17, 1, 11, 10, 0, 48, 28, 32, 35, 15, 28, 10, 48, 21, 8, 10, 29, 6, 33, 2, 10, 4, 33, 48, 33, 48, 49, 7, 46, 38, 33, 18, 10, 38, 9, 38, 33, 9, 35, 15, 46, 38, 15, 24, 10, 37, 44, 21, 0, 24, 10, 38, 37, 33, 40, 48, 48, 21, 31, 37, 43, 33, 46, 13, 42, 32, 46, 48, 48, 19, 48, 45, 0, 31, 1, 49, 3, 0, 49, 46, 1, 10, 38, 21, 48, 26, 15, 41, 9, 10, 1, 37, 16, 46, 1, 48, 48, 29, 39, 2, 46, 28, 33, 1, 29, 1, 21, 36, 2, 48, 2, 1, 40, 32, 10, 20, 26, 43, 42, 11, 17, 44, 43, 10, 33, 41, 33, 7, 48, 13, 35, 10, 16, 46, 18, 2, 46, 8, 0, 10, 1, 43, 30, 10, 38, 6, 33, 36, 15, 15, 6, 19, 33, 42, 45, 48, 10, 9, 35, 10, 26, 48, 48, 43, 48, 45, 33, 11, 30, 48, 33, 48, 36, 1, 11, 49, 39, 46, 42, 48, 43, 0, 33, 22, 48, 18, 1, 10, 6, 42, 33, 3, 10, 19, 35, 48, 9, 21, 40, 48, 33, 18, 0, 48, 10, 15, 11, 32, 16, 10, 7, 46, 48, 0, 43, 32, 10, 10, 48, 10, 33, 16, 26, 21, 7, 32, 48, 37, 0, 10, 48, 11, 30, 0, 43, 2, 48, 42, 30, 10, 33, 33, 32, 42, 48, 36, 19, 17, 15, 48, 27, 2, 15, 10, 19, 41, 35, 10, 40, 38, 10, 39, 39, 48, 38, 48, 32, 2, 39, 21, 2, 10, 15, 32, 48, 10, 1, 46, 1, 42, 32, 48, 10, 29, 27, 40, 2, 38, 16, 48, 20, 33, 8, 37, 44, 40, 16, 0, 33, 22, 48, 38, 0, 11, 5, 32, 24, 42, 48, 10, 9, 10, 5, 10, 48, 18, 30, 2, 16, 43, 0, 1, 16, 10, 45, 18, 48, 48, 2, 0, 33, 30, 48, 1, 0, 32, 33, 48, 10, 19, 33, 12, 6, 35, 16, 49, 10, 41, 0, 27, 30, 21, 15, 38, 2, 0, 48, 38, 11, 36, 0, 45, 24, 45, 49, 1, 18, 10, 36, 37, 30, 1, 48, 10, 23, 10, 1, 49, 40, 33, 48, 15, 43, 0, 33, 48, 29, 10, 0, 31, 10, 27, 48, 44, 18, 10, 48, 2, 10, 15, 48, 22, 10, 46, 43, 32, 19, 10, 4, 10, 0, 48, 13, 30, 2, 19, 30, 10, 33, 18, 10, 15, 48, 42, 36, 47, 45, 33, 41, 11, 48, 35, 15, 49, 19, 11, 48, 46, 10, 15, 48, 30, 21, 38, 27, 16, 4, 33, 5, 13, 40, 38, 0, 10, 45, 1, 1, 27, 18, 10, 38, 10, 1, 8, 10, 32, 8, 0, 15, 7, 0, 27, 10, 46, 4, 38, 12, 48, 38, 10, 48, 48, 36, 36, 4, 39, 48, 10, 33, 10, 48, 23, 1, 2, 2, 10, 40, 18, 0, 10, 0, 9, 3, 46, 4, 18, 10, 33, 2, 34, 35, 21, 48, 10, 33, 48, 0, 10, 0, 7, 0, 38, 30, 1, 10, 40, 46, 44, 43, 48, 3, 38, 23, 15, 29, 40, 48, 44, 47, 41, 46, 0, 39, 0, 21, 39, 46, 12, 7, 38, 48, 22, 33, 0, 30, 29, 41, 48, 41, 15, 45, 37, 18, 5, 18, 21, 10, 17, 17, 48, 48, 1, 46, 4, 38, 29, 48, 42, 30, 27, 36, 40, 1, 15, 10, 49, 19, 42, 20, 35, 33, 33, 2, 29, 16, 13, 48, 10, 48, 46, 18, 37, 0, 31, 43, 9, 15, 18, 27, 15, 20, 10, 44, 26, 48, 42, 48, 31, 28, 2, 40, 21, 10, 35, 46, 48, 38, 1, 43, 42, 35, 36, 37, 8, 10, 42, 48, 8, 12, 44, 42, 21, 43, 2, 48, 19, 46, 0, 37, 21, 27, 33, 48, 47, 21, 36, 41, 48, 33, 48, 32, 48, 33, 48, 35, 7, 10, 18, 49, 11, 44, 33, 47, 33, 10, 48, 42, 19, 30, 49, 24, 25, 42, 13, 5, 30, 7, 15, 48, 23, 29, 48, 9, 26, 28, 10, 40, 2, 35, 43, 0, 35, 24, 29, 27, 18, 24, 21, 43, 18, 41, 10, 10, 45, 43, 48, 33, 33, 48, 21, 16, 18, 12, 24, 45, 33, 48, 10, 42, 42, 48, 2, 35, 48, 2, 17, 10, 28, 32, 33, 30, 24, 45, 35, 2, 48, 37, 46, 18, 2, 22, 18, 37, 29, 46, 15, 1, 15, 44, 27, 36, 10, 30, 38, 4, 10, 10, 48, 0, 29, 40, 48, 25, 48, 29, 2, 4, 48, 38, 11, 38, 44, 15, 30, 39, 37, 44, 48, 0, 29, 21, 9, 23, 46, 48, 48, 18, 14, 48, 46, 32, 46, 16, 46, 48, 33, 30, 0, 33, 5, 48, 46, 48, 19, 14, 15, 33, 36, 42, 3, 38, 35, 10, 30, 43, 1, 48, 16, 9, 9, 15, 33, 5, 48, 19, 28, 24, 35, 33, 10, 18, 29, 38, 27, 16, 0, 10, 47, 33, 19, 40, 41, 48, 19, 39, 10, 45, 18, 24, 39, 15, 47, 16, 9, 36, 33, 30, 35, 48, 0, 37, 41, 40, 19, 33, 0, 46, 10, 43, 42, 48, 4, 32, 33, 32, 36, 2, 46, 12, 0, 48, 33, 41, 15, 48, 28, 30, 17, 11, 1, 10, 9, 10, 11, 2, 43, 33, 43, 22, 4, 7, 10, 34, 10, 35, 43, 48, 49, 21, 49, 43, 4, 33, 48, 48, 42, 1, 48, 49, 36, 1, 7, 10, 15, 13, 16, 7, 18, 4, 5, 9, 24, 15, 0, 10, 1, 15, 44, 18, 10, 10, 21, 48, 38, 4, 35, 10, 21, 32, 40, 35, 10, 31, 0, 13, 33, 29, 48, 23, 15, 17, 40, 21, 48, 21, 33, 35, 38, 20, 37, 22, 38, 35, 18, 2, 15, 10, 32, 18, 32, 16, 9, 45, 48, 46, 48, 41, 38, 0, 35, 22, 46, 48, 32, 10, 33, 16, 24, 45, 48, 48, 18, 33, 0, 27, 40, 0, 35, 2, 0, 28, 30, 19, 36, 10, 1, 38, 10, 46, 15, 0, 1, 36, 46, 22, 1, 48, 45, 43, 7, 18, 4, 3, 2, 1, 30, 33, 32, 49, 10, 42, 0, 41, 30, 38, 35, 16, 30, 9, 7, 47, 2, 35, 10, 20, 33, 0, 48, 44, 38, 0, 30, 11, 31, 18, 48, 37, 28, 18, 10, 38, 33, 48, 33, 15, 43, 48, 47, 42, 42, 37, 9, 44, 10, 15, 3, 48, 48, 2, 39, 15, 7, 46, 26, 17, 9, 0, 0, 46, 48, 33, 23, 15, 46, 19, 38, 10, 48, 1, 45, 16, 0, 33, 10, 22, 10, 18, 3, 10, 33, 9, 4, 24, 39, 44, 46, 45, 10, 34, 43, 10, 33, 6, 1, 33, 38, 28, 48, 46, 10, 42, 33, 18, 1, 2, 36, 21, 46, 45, 15, 48, 33, 0, 33, 40, 43, 49, 7, 10, 27, 47, 0, 32, 4, 30, 38, 18, 32, 21, 46, 40, 28, 48, 17, 2, 32, 5, 3, 19, 0, 1, 35, 11, 48, 15, 43, 2, 6, 29, 18, 47, 33, 3, 0, 18, 33, 30, 10, 4, 38, 35, 30, 43, 7, 33, 46, 11, 45, 42, 31, 46, 33, 12, 33, 48, 10, 33, 16, 26, 41, 46, 33, 32, 12, 33, 38, 4, 38, 33, 15, 48, 28, 42, 43, 21, 10, 15, 4, 32, 18, 35, 35, 0, 48, 43, 48, 1, 21, 48, 35, 30, 35, 37, 43, 10, 34, 32, 21, 2, 36, 38, 16, 0, 18, 10, 48, 21, 22, 42, 35, 32, 38, 3, 42, 45, 19, 48, 48, 26, 5, 10, 33, 49, 38, 17, 48, 33, 10, 42, 35, 7, 29, 38, 33, 2, 48, 46, 2, 2, 38, 10, 0, 36, 30, 48, 12, 0, 38, 10, 47, 48, 2, 10, 15, 48, 0, 9, 38, 19, 10, 35, 19, 28, 33, 10, 2, 42, 10, 3, 16, 10, 21, 30, 15, 36, 0, 15, 2, 0, 22, 48, 48, 22, 46, 38, 44, 45, 21, 41, 34, 48, 7, 37, 23, 10, 37, 10, 48, 18, 18, 18, 13, 2, 48, 48, 35, 5, 49, 36, 10, 1, 10, 31, 0, 38, 38, 12, 48, 33, 48, 1, 5, 38, 7, 5, 10, 6, 7, 42, 1, 37, 39, 10, 43, 21, 16, 4, 8, 35, 46, 10, 21, 11, 48, 12, 15, 10, 9, 45, 1, 37, 48, 2, 45, 0, 2, 27, 38, 33, 0, 3, 48, 5, 8, 35, 45, 35, 30, 45, 1, 21, 37, 31, 48, 36, 16, 0, 30, 5, 18, 42, 19, 37, 48, 48, 15, 10, 14, 35, 3, 40, 48, 27, 37, 2, 1, 45, 48, 43, 28, 33, 2, 18, 27, 22, 33, 48, 48, 15, 38, 26, 48, 48, 48, 48, 40, 38, 2, 10, 38, 10, 7, 48, 45, 48, 19, 8, 15, 38, 10, 33, 6, 10, 10, 26, 48, 2, 41, 1, 43, 33, 10, 10, 12, 0, 16, 15, 10, 1, 10, 48, 27, 33, 48, 11, 30, 35, 46, 18, 10, 36, 38, 16, 21, 43, 8, 49, 48, 10, 1, 1, 35, 48, 7, 4, 48, 44, 40, 30, 5, 15, 30, 33, 10, 3, 48, 35, 48, 0, 48, 48, 13, 46, 46, 46, 15, 48, 49, 46, 17, 30, 27, 37, 44, 1, 39, 10, 48, 23, 41, 45, 48, 18, 2, 10, 35, 35, 10, 0, 39, 49, 38, 2, 32, 30, 46, 48, 33, 37, 32, 22, 38, 32, 5, 10, 28, 16, 0, 10, 10, 18, 32, 2, 30, 42, 0, 12, 8, 33, 15, 38, 13, 10, 47, 48, 48, 9, 33, 38, 45, 24, 10, 48, 46, 12, 31, 0, 0, 10, 43, 11, 24, 30, 36, 2, 8, 18, 48, 27, 26, 21, 7, 48, 8, 11, 48, 1, 33, 35, 35, 48, 10, 33, 33, 18, 4, 36, 48, 34, 10, 10, 38, 13, 4, 4, 2, 4, 1, 10, 27, 48, 38, 48, 1, 38, 44, 27, 43, 33, 35, 1, 48, 48, 38, 0, 0, 44, 4, 41, 43, 30, 20, 15, 3, 45, 7, 0, 30, 15, 10, 35, 48, 37, 2, 16, 45, 18, 48, 4, 1, 42, 10, 48, 36, 48, 4, 1, 33, 8, 1, 44, 34, 33, 38, 30, 48, 29, 37, 21, 33, 46, 33, 48, 43, 1, 38, 12, 48, 48, 44, 38, 48, 36, 35, 37, 10, 0, 45, 1, 10, 48, 44, 38, 48, 15, 24, 45, 10, 35, 35, 10, 36, 30, 3, 28, 10, 9, 48, 39, 19, 3, 36, 15, 7, 46, 18, 43, 48, 6, 38, 15, 7, 45, 4, 23, 30, 22, 48, 5, 8, 20, 21, 18, 48, 22, 48, 33, 48, 33, 21, 1, 41, 33, 0, 19, 24, 38, 21, 38, 47, 1, 44, 32, 48, 35, 10, 8, 0, 48, 47, 4, 44, 10, 43, 10, 42, 10, 8, 0, 42, 10, 18, 48, 15, 6, 38, 19, 0, 33, 38, 41, 49, 43, 40, 1, 38, 35, 42, 33, 15, 45, 30, 48, 46, 15, 0, 30, 33, 35, 37, 48, 37, 10, 48, 43, 30, 10, 48, 46, 10, 2, 40, 41, 0, 33, 44, 0, 33, 44, 30, 42, 23, 43, 34, 10, 30, 43, 2, 23, 16, 48, 9, 48, 24, 48, 33, 42, 43, 10, 10, 46, 3, 19, 15, 2, 7, 45, 16, 18, 40, 35, 29, 36, 42, 38, 0, 19, 37, 48, 1, 10, 23, 0, 16, 2, 1, 0, 23, 37, 48, 48, 42, 46, 27, 29, 38, 48, 0, 33, 13, 9, 35, 16, 19, 40, 35, 45, 40, 15, 10, 0, 22, 44, 33, 37, 48, 39, 46, 48, 48, 16, 41, 7, 4, 18, 2, 30, 5, 30, 17, 48, 3, 38, 13, 48, 0, 2, 23, 48, 48, 7, 46, 41, 33, 46, 20, 21, 1, 40, 18, 48, 6, 42, 48, 46, 10, 9, 48, 30, 19, 46, 48, 35, 41, 38, 48, 23, 1, 21, 33, 48, 48, 36, 48, 24, 10, 16, 0, 47, 9, 13, 30, 16, 10, 45, 1, 35, 39, 10, 33, 33, 10, 13, 10, 19, 29, 35, 42, 0, 35, 46, 30, 48, 0, 41, 4, 35, 30, 30, 10, 10, 33, 11, 44, 48, 18, 43, 43, 0, 33, 35, 44, 43, 43, 29, 48, 24, 48, 21, 32, 43, 30, 39, 42, 11, 48, 33, 1, 48, 46, 10, 48, 43, 21, 36, 0, 46, 15, 32, 49, 21, 27, 36, 40, 10, 17, 30, 38, 32, 48, 21, 18, 29, 19, 15, 10, 10, 16, 5, 4, 19, 28, 38, 32, 10, 48, 41, 33, 28, 0, 42, 8, 46, 12, 35, 48, 31, 48, 11, 49, 48, 38, 48, 48, 1, 48, 28, 42, 10, 9, 49, 21, 6, 0, 48, 18, 24, 35, 32, 0, 16, 48, 16, 15, 36, 15, 21, 4, 38, 0, 18, 10, 30, 48, 3, 10, 13, 1, 46, 22, 4, 11, 40, 30, 43, 48, 42, 10, 1, 2, 2, 38, 38, 30, 10, 8, 10, 10, 10, 10, 10, 1, 10, 43, 38, 15, 38, 37, 30, 35, 10, 42, 46, 32, 1, 1, 37, 35, 10, 29, 10, 43, 35, 45, 1, 48, 45, 28, 43, 48, 0, 19, 48, 48, 35, 10, 48, 42, 18, 40, 9, 8, 21, 39, 48, 46, 15, 35, 10, 10, 1, 36, 29, 11, 18, 45, 1, 33, 9, 49, 35, 1, 48, 40, 35, 40, 30, 10, 5, 18, 33, 2, 41, 17, 38, 18, 38, 32, 47, 2, 38, 42, 42, 15, 15, 48, 38, 29, 15, 48, 19, 0, 6, 46, 9, 42, 48, 10, 33, 44, 4, 16, 48, 43, 48, 0, 48, 42, 1, 15, 35, 45, 19, 43, 32, 30, 1, 48, 13, 2, 43, 48, 32, 21, 1, 48, 40, 40, 38, 24, 48, 0, 45, 19, 33, 16, 4, 7, 8, 10, 35, 30, 38, 27, 33, 0, 10, 2, 3, 37, 41, 24, 17, 10, 28, 38, 29, 32, 33, 2, 15, 5, 45, 13, 47, 4, 49, 38, 48, 15, 6, 44, 48, 10, 34, 33, 42, 13, 38, 4, 48, 19, 10, 13, 8, 9, 48, 48, 33, 43, 48, 18, 19, 49, 10, 22, 10, 15, 27, 10, 19, 21, 0, 41, 28, 10, 37, 46, 48, 10, 0, 1, 46, 38, 39, 24, 48, 36, 15, 48, 44, 43, 48, 21, 46, 42, 35, 14, 35, 8, 10, 21, 33, 46, 0, 15, 44, 38, 35, 27, 33, 30, 48, 30, 8, 0, 21, 30, 48, 4, 15, 10, 44, 22, 18, 29, 42, 46, 33, 43, 48, 40, 10, 0, 29, 28, 45, 33, 2, 35, 42, 32, 10, 10, 48, 12, 34, 30, 4, 16, 43, 18, 47, 35, 2, 14, 2, 46, 42, 45, 32, 2, 18, 37, 35, 42, 0, 10, 30, 47, 1, 35, 10, 10, 10, 33, 1, 35, 38, 44, 35, 38, 33, 23, 44, 36, 46, 6, 49, 0, 30, 48, 18, 35, 33, 31, 12, 2, 19, 14, 48, 48, 10, 35, 44, 38, 34, 15, 46, 7, 27, 15, 48, 35, 4, 48, 21, 43, 21, 45, 48, 46, 10, 48, 38, 33, 0, 10, 5, 38, 33, 38, 48, 4, 10, 15, 2, 8, 48, 3, 45, 48, 10, 8, 21, 19, 46, 42, 46, 0, 35, 21, 35, 48, 18, 0, 10, 46, 48, 15, 10, 12, 0, 33, 10, 15, 27, 16, 42, 48, 48, 48, 43, 9, 2, 24, 48, 1, 31, 42, 48, 15, 10, 44, 10, 15, 37, 28, 43, 19, 31, 3, 48, 10, 15, 31, 15, 30, 10, 44, 35, 46, 48, 35, 42, 6, 48, 15, 30, 42, 48, 46, 48, 8, 43, 2, 0, 17, 33, 1, 16, 5, 35, 45, 15, 38, 28, 28, 48, 19, 35, 38, 16, 45, 31, 4, 46, 34, 42, 47, 48, 15, 38, 28, 1, 33, 19, 42, 37, 43, 43, 29, 40, 0, 38, 18, 1, 3, 38, 10, 32, 10, 10, 45, 15, 10, 30, 38, 37, 48, 10, 8, 0, 21, 35, 6, 19, 43, 36, 30, 10, 48, 16, 16, 33, 40, 43, 2, 35, 30, 35, 13, 4, 32, 10, 21, 1, 44, 37, 46, 15, 0, 18, 0, 30, 37, 10, 15, 48, 10, 45, 18, 32, 36, 46, 6, 15, 9, 15, 0, 18, 2, 2, 10, 38, 41, 29, 42, 38, 2, 3, 46, 1, 21, 33, 33, 9, 12, 33, 21, 13, 32, 36, 16, 36, 24, 12, 24, 11, 5, 10, 48, 15, 1, 1, 48, 46, 27, 43, 48, 49, 3, 5, 10, 33, 8, 48, 18, 24, 8, 48, 10, 3, 40, 1, 10, 45, 4, 48, 36, 19, 18, 46, 35, 48, 46, 48, 48, 48, 15, 33, 1, 33, 42, 33, 43, 2, 48, 10, 30, 15, 33, 41, 40, 48, 33, 37, 40, 40, 20, 35, 16, 31, 41, 29, 22, 40, 28, 30, 16, 44, 29, 40, 0, 48, 46, 48, 48, 48, 31, 38, 42, 46, 33, 2, 28, 33, 21, 0, 49, 48, 48, 48, 43, 15, 33, 28, 36, 32, 11, 2, 15, 35, 8, 1, 30, 30, 1, 39, 20, 10, 48, 18, 9, 21, 32, 33, 21, 29, 48, 35, 17, 7, 24, 43, 45, 49, 10, 31, 44, 47, 2, 42, 16, 42, 48, 18, 48, 33, 35, 33, 34, 7, 28, 13, 33, 44, 17, 21, 48, 7, 48, 38, 13, 1, 43, 1, 20, 39, 19, 0, 4, 38, 29, 48, 10, 29, 42, 35, 13, 1, 21, 1, 14, 22, 26, 36, 47, 33, 35, 38, 40, 45, 45, 3, 1, 41, 21, 33, 36, 0, 21, 35, 21, 16, 46, 21, 0, 1, 10, 48, 29, 5, 38, 12, 21, 42, 38, 14, 48, 42, 32, 33, 48, 47, 9, 0, 46, 35, 43, 48, 4, 10, 44, 33, 48, 47, 40, 42, 36, 0, 38, 33, 36, 42, 7, 4, 35, 49, 32, 21, 10, 35, 33, 0, 45, 37, 44, 25, 45, 28, 37, 18, 48, 48, 10, 48, 18, 44, 10, 48, 48, 42, 45, 3, 2, 48, 7, 48, 32, 10, 19, 10, 48, 35, 48, 33, 13, 48, 17, 48, 10, 38, 48, 10, 0, 10, 32, 18, 33, 36, 38, 42, 48, 10, 48, 42, 2, 2, 26, 10, 7, 24, 48, 18, 10, 10, 23, 9, 10, 38, 43, 11, 15, 21, 48, 0, 33, 37, 0, 13, 10, 2, 1, 42, 45, 12, 5, 7, 33, 3, 46, 23, 1, 0, 0, 38, 29, 10, 1, 35, 1, 19, 1, 26, 1, 38, 42, 38, 10, 48, 18, 44, 40, 47, 34, 10, 48, 19, 0, 27, 6, 10, 28, 15, 4, 46, 38, 46, 33, 1, 11, 2, 22, 36, 34, 32, 32, 45, 42, 33, 45, 21, 10, 38, 4, 35, 34, 28, 48, 1, 35, 35, 35, 17, 6, 42, 46, 2, 4, 26, 38, 32, 5, 10, 10, 33, 1, 46, 24, 46, 38, 0, 15, 10, 48, 48, 13, 44, 46, 2, 40, 43, 45, 32, 10, 38, 15, 45, 45, 33, 21, 18, 45, 27, 10, 10, 37, 27, 0, 8, 48, 10, 41, 2, 0, 38, 44, 2, 7, 12, 46, 2, 4, 33, 18, 48, 34, 10, 9, 33, 10, 7, 10, 46, 0, 40, 46, 22, 28, 21, 29, 1, 10, 34, 32, 28, 42, 1, 48, 35, 18, 13, 46, 48, 10, 19, 22, 2, 33, 4, 6, 0, 1, 48, 13, 38, 40, 48, 46, 13, 19, 2, 49, 28, 38, 48, 33, 2, 4, 11, 36, 21, 1, 46, 15, 9, 49, 41, 45, 16, 10, 10, 48, 48, 38, 13, 32, 26, 28, 10, 10, 33, 23, 42, 48, 42, 37, 35, 31, 1, 43, 48, 47, 48, 33, 11, 48, 23, 10, 0, 9, 16, 21, 9, 48, 10, 45, 8, 46, 37, 48, 3, 22, 33, 24, 15, 1, 32, 48, 1, 48, 33, 10, 10, 12, 28, 33, 4, 10, 10, 15, 5, 15, 0, 48, 4, 46, 2, 19, 46, 11, 41, 46, 0, 12, 40, 15, 48, 2, 19, 0, 0, 1, 45, 48, 48, 21, 27, 42, 35, 21, 40, 47, 48, 13, 33, 15, 26, 21, 37, 42, 40, 46, 23, 2, 33, 48, 2, 6, 33, 48, 37, 46, 35, 46, 29, 44, 48, 30, 16, 33, 35, 28, 42, 1, 12, 3, 26, 16, 32, 21, 15, 48, 15, 15, 38, 29, 0, 10, 48, 4, 40, 48, 1, 48, 28, 0, 2, 38, 13, 48, 36, 0, 2, 45, 10, 1, 0, 33, 38, 46, 32, 42, 10, 18, 32, 33, 21, 42, 45, 46, 37, 44, 48, 35, 47, 21, 10, 42, 48, 38, 46, 32, 1, 35, 37, 9, 29, 9, 2, 46, 42, 10, 10, 1, 43, 46, 10, 17, 11, 30, 21, 16, 28, 1, 35, 38, 40, 21, 32, 2, 46, 10, 0, 27, 15, 2, 48, 38, 36, 21, 27, 19, 30, 24, 1, 11, 1, 43, 0, 38, 1, 15, 20, 45, 1, 10, 37, 48, 13, 1, 33, 15, 45, 38, 32, 3, 10, 1, 46, 15, 48, 28, 8, 35, 10, 12, 15, 48, 10, 15, 47, 41, 2, 38, 21, 48, 48, 0, 20, 43, 28, 48, 33, 36, 1, 46, 48, 30, 22, 29, 35, 42, 2, 44, 1, 10, 1, 1, 48, 38, 30, 12, 33, 10, 38, 10, 5, 0, 45, 42, 28, 45, 49, 10, 21, 9, 22, 29, 42, 6, 43, 15, 32, 1, 48, 48, 48, 43, 39, 48, 46, 46, 48, 48, 35, 0, 45, 8, 48, 1, 30, 15, 7, 33, 40, 22, 48, 0, 41, 7, 18, 30, 6, 42, 16, 28, 21, 10, 12, 48, 48, 48, 19, 2, 48, 31, 15, 15, 48, 40, 42, 29, 10, 38, 16, 44, 43, 38, 12, 7, 43, 5, 48, 0, 48, 30, 30, 0, 26, 15, 2, 38, 21, 48, 15, 37, 43, 2, 8, 30, 38, 3, 8, 10, 36, 1, 1, 21, 10, 13, 48, 0, 47, 30, 39, 2, 10, 28, 39, 21, 48, 33, 2, 35, 32, 9, 44, 4, 24, 4, 18, 10, 19, 1, 48, 15, 29, 2, 15, 37, 28, 48, 49, 0, 19, 46, 37, 15, 44, 33, 46, 1, 32, 1, 9, 38, 10, 16, 32, 11, 1, 33, 8, 35, 1, 48, 37, 48, 8, 36, 24, 48, 38, 42, 28, 35, 41, 1, 48, 48, 32, 33, 17, 31, 48, 35, 4, 44, 48, 10, 37, 16, 48, 32, 18, 41, 46, 33, 33, 48, 47, 38, 48, 13, 48, 38, 48, 47, 0, 15, 9, 32, 37, 15, 33, 48, 1, 26, 38, 46, 29, 40, 48, 48, 32, 38, 10, 48, 48, 42, 26, 15, 38, 46, 15, 40, 15, 10, 10, 10, 38, 11, 48, 46, 1, 33, 6, 10, 7, 10, 37, 7, 42, 15, 30, 48, 18, 28, 15, 29, 18, 48, 0, 45, 1, 16, 46, 38, 48, 41, 15, 33, 35, 42, 21, 36, 10, 15, 29, 16, 1, 32, 43, 36, 48, 48, 1, 10, 34, 10, 45, 10, 8, 7, 48, 44, 28, 46, 44, 43, 36, 45, 24, 16, 46, 38, 0, 0, 15, 10, 10, 27, 18, 10, 1, 28, 35, 48, 45, 21, 41, 4, 38, 18, 34, 15, 23, 48, 38, 43, 35, 39, 13, 15, 36, 35, 10, 48, 30, 1, 48, 15, 21, 36, 0, 45, 7, 6, 48, 19, 1, 2, 44, 10, 48, 10, 3, 37, 33, 1, 15, 1, 48, 10, 12, 0, 45, 29, 40, 45, 43, 33, 48, 38, 16, 45, 46, 10, 45, 38, 40, 18, 48, 9, 45, 16, 48, 28, 10, 40, 10, 31, 38, 48, 10, 0, 42, 40, 42, 12, 1, 23, 33, 36, 43, 43, 23, 15, 2, 1, 33, 48, 48, 37, 19, 15, 48, 27, 35, 24, 15, 10, 42, 10, 33, 42, 42, 41, 32, 17, 33, 48, 4, 27, 36, 19, 21, 45, 0, 32, 10, 44, 48, 46, 35, 1, 17, 48, 23, 7, 0, 22, 48, 35, 10, 48, 10, 1, 43, 10, 21, 0, 47, 48, 45, 38, 29, 15, 48, 21, 43, 48, 33, 48, 38, 37, 0, 37, 46, 33, 12, 46, 30, 10, 15, 35, 9, 42, 15, 9, 10, 33, 0, 35, 44, 15, 2, 4, 21, 3, 46, 48, 4, 33, 18, 38, 9, 45, 16, 42, 38, 33, 1, 40, 34, 10, 2, 10, 10, 10, 10, 16, 23, 1, 45, 1, 0, 15, 43, 10, 44, 10, 4, 2, 43, 18, 10, 19, 0, 48, 38, 33, 43, 10, 47, 1, 37, 46, 23, 43, 4, 10, 48, 2, 42, 9, 15, 28, 28, 33, 5, 33, 33, 46, 30, 19, 26, 19, 5, 46, 0, 37, 37, 41, 35, 2, 45, 34, 33, 0, 10, 3, 2, 19, 0, 35, 33, 22, 28, 38, 0, 29, 23, 35, 32, 15, 48, 25, 15, 3, 48, 11, 37, 45, 30, 10, 10, 48, 38, 0, 35, 7, 10, 42, 15, 1, 17, 2, 21, 2, 10, 26, 6, 37, 48, 4, 35, 33, 48, 48, 2, 30, 30, 38, 0, 33, 23, 21, 38, 46, 43, 35, 4, 48, 0, 10, 18, 48, 32, 33, 36, 1, 44, 2, 15, 16, 28, 15, 21, 22, 48, 33, 36, 15, 38, 22, 21, 15, 10, 48, 40, 37, 48, 32, 43, 43, 3, 43, 2, 42, 29, 40, 15, 32, 19, 0, 18, 6, 10, 21, 42, 30, 12, 24, 45, 28, 48, 30, 48, 40, 31, 0, 10, 49, 42, 33, 28, 35, 27, 42, 10, 48, 0, 24, 7, 8, 29, 30, 3, 10, 21, 15, 33, 33, 28, 13, 3, 0, 37, 7, 46, 4, 0, 35, 15, 48, 3, 6, 21, 0, 44, 48, 0, 10, 4, 10, 38, 7, 42, 18, 3, 10, 35, 2, 33, 15, 19, 10, 8, 45, 9, 5, 17, 16, 17, 49, 6, 43, 9, 8, 33, 48, 46, 0, 19, 46, 48, 19, 10, 15, 0, 3, 0, 20, 46, 0, 42, 32, 43, 38, 4, 48, 21, 0, 42, 39, 10, 10, 10, 30, 31, 42, 10, 26, 38, 40, 10, 33, 35, 39, 19, 48, 42, 3, 46, 48, 16, 2, 4, 10, 10, 9, 2, 46, 48]
    

# Make Submission


```python
preds = le.inverse_transform(result) # LabelEncoder로 변환 된 Label을 다시 화가이름으로 변환
```


```python
print(le.classes_)
```

    ['Albrecht Du rer' 'Alfred Sisley' 'Amedeo Modigliani' 'Andrei Rublev'
     'Andy Warhol' 'Camille Pissarro' 'Caravaggio' 'Claude Monet'
     'Diego Rivera' 'Diego Velazquez' 'Edgar Degas' 'Edouard Manet'
     'Edvard Munch' 'El Greco' 'Eugene Delacroix' 'Francisco Goya'
     'Frida Kahlo' 'Georges Seurat' 'Giotto di Bondone' 'Gustav Klimt'
     'Gustave Courbet' 'Henri Matisse' 'Henri Rousseau'
     'Henri de Toulouse-Lautrec' 'Hieronymus Bosch' 'Jackson Pollock'
     'Jan van Eyck' 'Joan Miro' 'Kazimir Malevich' 'Leonardo da Vinci'
     'Marc Chagall' 'Michelangelo' 'Mikhail Vrubel' 'Pablo Picasso'
     'Paul Cezanne' 'Paul Gauguin' 'Paul Klee' 'Peter Paul Rubens'
     'Pierre-Auguste Renoir' 'Piet Mondrian' 'Pieter Bruegel' 'Raphael'
     'Rembrandt' 'Rene Magritte' 'Salvador Dali' 'Sandro Botticelli' 'Titian'
     'Vasiliy Kandinskiy' 'Vincent van Gogh' 'William Turner']
    


```python
submit = pd.read_csv('./submit.csv')
```


```python
submit['artist'] = preds
submit.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>artist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_00000</td>
      <td>Edgar Degas</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_00001</td>
      <td>Amedeo Modigliani</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_00002</td>
      <td>Leonardo da Vinci</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TEST_00003</td>
      <td>Albrecht Du rer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TEST_00004</td>
      <td>Edgar Degas</td>
    </tr>
  </tbody>
</table>
</div>




```python
submit.to_csv('submit2022-11-11.csv', index=False)
```
