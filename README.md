# Detectarea cancerului de sân cu YOLOv8

## Descriere
Acest proiect prezintă un workflow complet pentru antrenarea unui model YOLOv8 „nano” pe imagini ecografice de sân, cu scopul de a detecta tumori benigne și maligne. Pipeline-ul acoperă pregătirea datelor, antrenarea modelului, evaluarea metricilor și generarea de vizualizări pentru interpretarea rezultatelor.

## Structura proiectului
```plaintext
.
├── data/
│   ├── images_raw/       # Imagini originale .png
│   ├── labels_raw/       # Etichete YOLO (.txt)
│   ├── train/            # 80% imagini pentru antrenare
│   └── val/              # 20% imagini pentru validare
├── runs/
│   └── train/            # Loguri și rezultate de antrenare
├── notebooks/
│   ├── prepare_data.ipynb
│   ├── train_model.ipynb
│   └── evaluate.ipynb
├── models/
│   └── best.pt           # Modelul cu cea mai bună performanță
├── results/
│   ├── examples/         # Imagini rezultate cu bounding-box
│   ├── confusion_matrix.png
│   ├── results.png       # Grafice generale (loss, mAP)
│   ├── F1_curve.png
│   └── PR_curve.png
├── data.yaml             # Configurație date pentru YOLOv8
└── README.md


## Cerințe
- Python 3.8+
- Kaggle / Colab / mediu local cu GPU recomandat
- Biblioteca [ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8)  
  ```bash
  pip install ultralytics
  ``` :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
- Alte biblioteci:
  - `os`, `random`, `shutil`
  - `matplotlib`, `PIL`

## Pregătirea datelor
1. **Maparea claselor**  
   ```python
   class_mapping = {
     "benign": 0,    # tumori necanceroase
     "malignant": 1, # tumori canceroase
     "normal": 2     # țesut sănătos
   }
   ``` :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}

2. **Structura YOLO-style**  
   - Fiecare imagine .png fără sufixul “mask” este copiată în `images_raw/`.  
   - Pentru fiecare imagine, se creează un `.txt` cu formatul:  
     ```
     <class_id> <x_center> <y_center> <width> <height>
     ```
     (în acest proiect, bounding-box-ul acoperă întreaga imagine) :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}

3. **Împărțirea setului**  
   - 80% antrenare: `data/train/`  
   - 20% validare: `data/val/`

## Configurație YOLOv8 (`data.yaml`)
```yaml
path: ../data
train: images/train
val: images/val
nc: 3
names: ['benign', 'malignant', 'normal']

## Antrenarea
yolo detect train \
  model=yolov8n.pt \
  data=data.yaml \
  epochs=20 \
  imgsz=640 \
  batch=8
``` :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}

- **epochs**: 20  
- **imgsz**: 640  
- **batch**: 8  

### Monitorizare loss
Primele 10 epoci arată o scădere rapidă a `box_loss`, indicând convergență corectă. În ultimele 10 epoci, `box_loss` ajunge la ~0.10 :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}.

## Evaluare și metrici
La finalul antrenării, s-au calculat următoarele metrici:
```python
results = {
  'precision(B)': 0.34,    # 34%
  'recall(B)':    0.96,    # 96%
  'mAP50(B)':     0.43,    # 43%
  'mAP50-95(B)':  0.43,    # 43%
  'fitness':      0.43
}
``` :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}

- **Precision** scăzută (34%) → multe fals pozitive  
- **Recall** ridicat (96%) → detectează majoritatea tumorilor  
- **mAP50** & **mAP50-95** moderate (43%)  
- **Fitness** total: 0.43

## Vizualizări
- **Imagini cu bounding-box** (`results/examples/`)  
- **Matrice de confuzie**: `results/confusion_matrix.png`  
- **Curbe de performanță** (`results.png`, `F1_curve.png`, `PR_curve.png`)  
- Evoluția `loss` și a mAP pe epoci (stabilizare ∼ 43%) :contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13}.

## Utilizare
După antrenare, modelul salvat în `models/best.pt` poate fi folosit astfel:
```bash
yolo detect predict \
  model=models/best.pt \
  source=path/to/images \
  conf=0.25 \
  save=true
