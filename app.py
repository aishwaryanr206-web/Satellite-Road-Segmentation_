import os, re, secrets, uuid
from pathlib import Path
import cv2, numpy as np, torch
from flask import Flask, abort, render_template, request, send_from_directory, flash, redirect, url_for
from imagery import read_rgb_image
from model import ResNetUNet
from vectorize import mask_to_vectors
from reporting import create_pdf_report
import rasterio

ROOT=Path(__file__).resolve().parent
UPLOAD=ROOT/"uploads"; OUTPUT=ROOT/"outputs"; WEIGHTS=ROOT/"weights"/"road_resunet.pth"
UPLOAD.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)
app=Flask(__name__)
app.secret_key=os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"]=100*1024*1024
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
model=None; image_size=256
threshold=0.5

def load_model():
    global model, image_size, threshold
    if not WEIGHTS.exists(): return False
    ck=torch.load(WEIGHTS,map_location=DEVICE)
    image_size=int(ck.get("image_size",256))
    threshold=float(ck.get("threshold",0.5))
    if not 0 <= threshold <= 1: raise ValueError("Checkpoint threshold must be between 0 and 1.")
    model=ResNetUNet(pretrained=False).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    return True

def predict(path, rgb=None):
    global model
    if model is None and not load_model(): raise RuntimeError("Model checkpoint not found. Train first using train.py; see README.")
    if rgb is None:
        rgb=read_rgb_image(path)
    height,width=rgb.shape[:2]
    pad_h=max(0,image_size-height); pad_w=max(0,image_size-width)
    if pad_h or pad_w:
        rgb=np.pad(rgb,((0,pad_h),(0,pad_w),(0,0)),mode="symmetric")
    height,width=rgb.shape[:2]
    stride=max(1,image_size//2)
    ys=list(range(0,max(1,height-image_size+1),stride))
    xs=list(range(0,max(1,width-image_size+1),stride))
    if ys[-1]!=height-image_size: ys.append(height-image_size)
    if xs[-1]!=width-image_size: xs.append(width-image_size)
    mean=torch.tensor([.485,.456,.406])[:,None,None]; std=torch.tensor([.229,.224,.225])[:,None,None]
    probability=np.zeros((height,width),dtype=np.float32)
    weight=np.zeros((height,width),dtype=np.float32)
    tiles=[(y,x) for y in ys for x in xs]
    with torch.no_grad():
        for start in range(0,len(tiles),4):
            batch_tiles=tiles[start:start+4]
            arrays=[rgb[y:y+image_size,x:x+image_size] for y,x in batch_tiles]
            x=torch.from_numpy(np.stack(arrays).astype(np.float32)/255.).permute(0,3,1,2)
            x=((x-mean)/std).to(DEVICE)
            probs=torch.sigmoid(model(x)).cpu().numpy()[:,0]
            for prob,(y,x0) in zip(probs,batch_tiles):
                probability[y:y+image_size,x0:x0+image_size]+=prob
                weight[y:y+image_size,x0:x0+image_size]+=1
    probability/=np.maximum(weight,1)
    mask=(probability[:rgb.shape[0]-pad_h or None,:rgb.shape[1]-pad_w or None]>=threshold).astype(np.uint8)*255
    return mask

def save_result_previews(rgb, mask, output_dir, max_dimension=1600):
    """Save browser-sized input and overlay previews while retaining the full-resolution mask."""
    height, width = mask.shape
    scale = min(1.0, max_dimension / max(height, width))
    preview_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if scale < 1:
        preview_rgb = cv2.resize(rgb, preview_size, interpolation=cv2.INTER_AREA)
        preview_mask = cv2.resize(mask, preview_size, interpolation=cv2.INTER_NEAREST)
    else:
        preview_rgb = rgb
        preview_mask = mask

    overlay = preview_rgb.copy()
    road_pixels = preview_mask > 0
    overlay[road_pixels] = (
        overlay[road_pixels].astype(np.float32) * 0.45
        + np.array([0, 255, 120], dtype=np.float32) * 0.55
    ).astype(np.uint8)

    for filename, image in (
        ("input_preview.png", preview_rgb),
        ("prediction_overlay.png", overlay),
    ):
        path = output_dir / filename
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write result preview: {path}")

@app.route("/",methods=["GET","POST"])
def index():
    if request.method=="POST":
        f=request.files.get("image")
        if not f or not f.filename: flash("Choose a satellite image first."); return redirect(url_for("index"))
        ext=Path(f.filename).suffix.lower()
        if ext not in {".png",".jpg",".jpeg",".tif",".tiff"}: flash("Supported files: PNG, JPG, JPEG, TIF, TIFF."); return redirect(url_for("index"))
        job=uuid.uuid4().hex[:10]; inp=UPLOAD/f"{job}{ext}"; f.save(inp)
        out=OUTPUT/job; out.mkdir(exist_ok=True)
        try:
            rgb=read_rgb_image(inp)
            mask=predict(inp,rgb=rgb); mask_path=out/"road_mask.png"
            if not cv2.imwrite(str(mask_path),mask):
                raise OSError(f"Could not write predicted mask: {mask_path}")
            save_result_previews(rgb,mask,out)
            georeference=None
            if Path(inp).suffix.lower() in {".tif",".tiff"}:
                with rasterio.open(inp) as source:
                    if source.crs is not None:
                        georeference=inp
            stats,geojson,kml=mask_to_vectors(mask_path,out,georeference=georeference)
            pdf=out/"report.pdf"; create_pdf_report(pdf, f.filename, stats)
            return render_template("result.html", job=job, stats=stats)
        except Exception as e:
            app.logger.exception("Road segmentation failed for job %s",job)
            flash(str(e)); return redirect(url_for("index"))
        finally:
            inp.unlink(missing_ok=True)
    return render_template("index.html", model_ready=WEIGHTS.exists())

@app.route("/files/<job>/<path:filename>")
def files(job,filename):
    if not re.fullmatch(r"[0-9a-f]{10}",job):
        abort(404)
    folder=OUTPUT/job
    if not folder.exists(): return "Not found",404
    return send_from_directory(folder,filename,as_attachment=True)

@app.route("/preview/<job>/<filename>")
def previews(job,filename):
    if not re.fullmatch(r"[0-9a-f]{10}",job):
        abort(404)
    if filename not in {"input_preview.png","prediction_overlay.png"}:
        abort(404)
    folder=OUTPUT/job
    if not folder.exists(): return "Not found",404
    return send_from_directory(folder,filename,as_attachment=False,mimetype="image/png",max_age=0)

@app.errorhandler(413)
def upload_too_large(_error):
    flash("Image exceeds the 100 MB upload limit.")
    return redirect(url_for("index")),413

if __name__=="__main__":
    app.run(host="127.0.0.1",port=5000,debug=os.environ.get("FLASK_DEBUG")=="1")
