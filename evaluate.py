import argparse, torch, numpy as np
from torch.utils.data import DataLoader
from train import RoadDataset
from model import ResNetUNet

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data_dir",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--image_size",type=int,default=256); ap.add_argument("--threshold",type=float); args=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    ck=torch.load(args.checkpoint,map_location=device)
    threshold=float(ck.get("threshold",.5) if args.threshold is None else args.threshold)
    if not 0 <= threshold <= 1:
        ap.error("--threshold must be between 0 and 1.")
    size=ck.get("image_size",args.image_size)
    ds=RoadDataset(args.data_dir,size); dl=DataLoader(ds,batch_size=1,shuffle=False)
    model=ResNetUNet(pretrained=False).to(device); model.load_state_dict(ck["model_state_dict"]); model.eval()
    tp=fp=fn=tn=0
    with torch.no_grad():
        for x,y in dl:
            p=(torch.sigmoid(model(x.to(device)))>=threshold).cpu().numpy().astype(bool)
            y=y.numpy().astype(bool)
            tp+=np.logical_and(p,y).sum(); fp+=np.logical_and(p,~y).sum()
            fn+=np.logical_and(~p,y).sum(); tn+=np.logical_and(~p,~y).sum()
    acc=(tp+tn)/max(1,tp+tn+fp+fn); precision=tp/max(1,tp+fp); dice=2*tp/max(1,2*tp+fp+fn); iou=tp/max(1,tp+fp+fn)
    print(f"Pixel Accuracy: {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"F1 / Dice: {dice*100:.2f}%")
    print(f"IoU: {iou*100:.2f}%")
    print(f"Threshold: {threshold:.2f}")
if __name__=="__main__": main()
