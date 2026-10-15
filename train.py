import argparse, os, random
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from tqdm import tqdm
from model import ResNetUNet, FocalTverskyLoss

class RoadDataset(Dataset):
    def __init__(self, root, image_size=256, augment=False):
        self.images = sorted(
            path for path in (Path(root)/"images").glob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )
        self.mask_dir = Path(root)/"masks"
        self.image_size = image_size
        self.augment = augment
        self.items = [p for p in self.images if (self.mask_dir/(p.stem+".png")).exists() or (self.mask_dir/(p.stem+".jpg")).exists()]
        if not self.items: raise RuntimeError("No image/mask pairs found. See README dataset layout.")
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p = self.items[i]
        m = next((self.mask_dir/(p.stem+ext) for ext in [".png",".jpg"] if (self.mask_dir/(p.stem+ext)).exists()))
        im = cv2.imread(str(p))
        mask = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        if im is None or mask is None:
            raise RuntimeError(f"Could not read image/mask pair: {p.name}, {m.name}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.image_size,self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size,self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            turns = random.randrange(4)
            im = np.rot90(im, turns)
            mask = np.rot90(mask, turns)
            if random.random() < 0.5:
                im = np.flip(im, axis=1)
                mask = np.flip(mask, axis=1)
            if random.random() < 0.5:
                im = np.flip(im, axis=0)
                mask = np.flip(mask, axis=0)
        x = torch.from_numpy(im.astype(np.float32)/255.).permute(2,0,1)
        mean = torch.tensor([.485,.456,.406])[:,None,None]
        std = torch.tensor([.229,.224,.225])[:,None,None]
        x = (x-mean)/std
        y = torch.from_numpy((mask>127).astype(np.float32))[None,:,:]
        return x.contiguous(),y.contiguous()

def select_threshold(model, dataloader, device):
    probabilities, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            probabilities.append(torch.sigmoid(model(x.to(device))).cpu().numpy().ravel())
            targets.append(y.numpy().ravel().astype(bool))
    probabilities = np.concatenate(probabilities)
    targets = np.concatenate(targets)
    thresholds = np.arange(0.05, 1.0, 0.01)
    scores = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        tp = np.logical_and(predicted, targets).sum()
        fp = np.logical_and(predicted, ~targets).sum()
        fn = np.logical_and(~predicted, targets).sum()
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(thresholds[int(np.argmax(scores))])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir",required=True); ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--batch_size",type=int,default=4); ap.add_argument("--image_size",type=int,default=256)
    ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--val_split",type=float,default=.2)
    ap.add_argument("--val_dir",default=None,help="Optional separate paired validation directory.")
    ap.add_argument("--num_workers",type=int,default=0)
    ap.add_argument("--output",default="weights/road_resunet.pth")
    args=ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32 or args.num_workers < 0:
        ap.error("epochs/batch_size must be positive, image_size must be >=32, and num_workers non-negative.")
    if args.val_dir is None and not 0 < args.val_split < 1:
        ap.error("--val_split must be between 0 and 1 when --val_dir is not provided.")
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    ds=RoadDataset(args.data_dir,args.image_size,augment=True)
    if args.val_dir:
        if len(ds) < 1:
            raise ValueError("Training dataset is empty.")
        train_ds=ds
        val_ds=RoadDataset(args.val_dir,args.image_size,augment=False)
    else:
        nval=max(1,int(len(ds)*args.val_split)); ntrain=len(ds)-nval
        if ntrain < 1: raise ValueError("Need at least 2 paired images for train/validation split.")
        val_source=RoadDataset(args.data_dir,args.image_size,augment=False)
        train_indices,val_indices=random_split(range(len(ds)),[ntrain,nval],generator=torch.Generator().manual_seed(42))
        train_ds=Subset(ds,train_indices.indices)
        val_ds=Subset(val_source,val_indices.indices)
    train_dl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    val_dl=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=ResNetUNet(pretrained=True).to(device)
    criterion=FocalTverskyLoss(); opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    best=1e9; os.makedirs(Path(args.output).parent,exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); total=0
        for x,y in tqdm(train_dl,desc=f"Epoch {epoch+1}/{args.epochs}"):
            x,y=x.to(device),y.to(device); opt.zero_grad()
            loss=criterion(model(x),y); loss.backward(); opt.step(); total+=loss.item()
        model.eval(); vtotal=0
        with torch.no_grad():
            for x,y in val_dl: vtotal+=criterion(model(x.to(device)),y.to(device)).item()
        vloss=vtotal/max(1,len(val_dl))
        print(f"epoch={epoch+1} train_loss={total/max(1,len(train_dl)):.4f} val_loss={vloss:.4f}")
        if vloss<best:
            best=vloss; torch.save({"model_state_dict":model.state_dict(),"image_size":args.image_size,"epoch":epoch+1,"validation_loss":vloss},args.output)
            print(f"Saved best checkpoint: {args.output}")
    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    checkpoint["threshold"] = select_threshold(model, val_dl, device)
    torch.save(checkpoint, args.output)
    print(f"Selected validation threshold: {checkpoint['threshold']:.2f}")
if __name__=="__main__": main()
