# CMMLoc coarse + multiple fine models

Pipeline นี้รัน **CMMLoc coarse เพียงครั้งเดียว** แล้วส่งรายการ top-k cells
ชุดเดียวกันให้ fine stage ของ:

- CMMLoc
- MNCL
- `my_model`

ไฟล์ `.pth` ของแต่ละ architecture จะไม่ถูกโหลดข้าม class กัน ตัว MNCL
จะรันใน Python subprocess แยกต่างหาก เพื่อป้องกัน package `models`
ของ MNCL ชนกับ package `models` ของ CMMLoc

ผลลัพธ์ถูกบันทึกเป็น:

```text
results/cmmloc_coarse_multi_fine/
├── cmmloc_retrievals.json
├── mncl_fine.json
└── comparison.json
```

## โครงสร้างไฟล์ที่ต้องมี

```text
cmmloc_coarse_other_model_fine/
├── data/
│   └── k360_30-10_scG_pd10_pc4_spY_all/
├── checkpoints/
│   ├── pointnet_acc0.86_lr1_p256_model.pth
│   └── k360_30-10_scG_pd10_pc4_spY_all/
│       ├── CMMLoc/
│       │   ├── coarse.pth
│       │   ├── fine.pth
│       │   └── prealign_*.pth
│       ├── MNCL/
│       │   └── fine.pth
│       └── my_model/
│           ├── fine.pth
│           └── prealign_*.pth
└── t5-large/
```

`data/`, `checkpoints/`, `t5-large/`, `third_party/MNCL/` และ `results/`
ถูก ignore จาก Git เพราะมีขนาดใหญ่หรือสร้างใหม่ได้

## เตรียมเครื่อง GPU

หลังจาก `git pull` repository นี้แล้ว ให้เปิด PowerShell ที่ root ของ repository:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_mncl.ps1
```

คำสั่งนี้ clone [official MNCL repository](https://github.com/dqliua/MNCL)
ไว้ที่ `third_party/MNCL` โดย source ของ MNCL จะไม่ถูก copy หรือ commit
รวมเข้า repository นี้

โมเดลที่เผยแพร่ใช้ text backbone คนละตัวกัน:

- CMMLoc และ `my_model`: T5-large ปกติ (`google-t5/t5-large`)
- MNCL: Flan-T5-large (`google/flan-t5-large`)

ห้ามใช้ Flan-T5 กับ checkpoint CMMLoc เพราะ architecture เหมือนกันจึงโหลด
ได้โดยไม่ error แต่ embeddings ไม่ตรงกับตอน train และ retrieval จะเกือบเป็นการสุ่ม
กำหนดแยกกันผ่าน `--cmmloc_t5_path` และ `--mncl_t5_path`

MNCL official implementation แนะนำ Python 3.10, PyTorch 1.11 และ CUDA
11.3 ดู environment เพิ่มเติมจาก official MNCL README

## รัน validation set

```powershell
python -m evaluation.hybrid_pipeline `
  --cmmloc_t5_path google-t5/t5-large `
  --mncl_t5_path google/flan-t5-large `
  --top_k 1 3 5 10 `
  --threshs 5 10 15
```

ค่า default ใช้:

```text
data/k360_30-10_scG_pd10_pc4_spY_all
checkpoints/k360_30-10_scG_pd10_pc4_spY_all
checkpoints/k360_30-10_scG_pd10_pc4_spY_all/CMMLoc/prealign_pointnet.pth
```

## รัน test set

```powershell
python -m evaluation.hybrid_pipeline `
  --cmmloc_t5_path google-t5/t5-large `
  --mncl_t5_path google/flan-t5-large `
  --use_test_set `
  --top_k 1 3 5 10 `
  --threshs 5 10 15
```

ตรวจ coarse อย่างเดียวก่อนเริ่ม fine stage ที่ใช้เวลานาน:

```powershell
python -m evaluation.hybrid_pipeline `
  --cmmloc_t5_path google-t5/t5-large `
  --use_test_set `
  --coarse_only `
  --top_k 1 3 5 10 `
  --threshs 5 10 15
```

เลือก fine models บางตัวได้ด้วย:

```powershell
python -m evaluation.hybrid_pipeline `
  --cmmloc_t5_path google-t5/t5-large `
  --fine_models CMMLoc my_model
```

## ใช้ path อื่น

ทุก path เป็น relative ต่อ root repository ได้ จึงใช้โค้ดเดียวกันได้ทั้งสองเครื่อง:

```powershell
python -m evaluation.hybrid_pipeline `
  --base_path .\data\k360_30-10_scG_pd10_pc4_spY_all `
  --checkpoint_root .\checkpoints\k360_30-10_scG_pd10_pc4_spY_all `
  --mncl_root .\third_party\MNCL `
  --cmmloc_t5_path .\t5-large `
  --mncl_t5_path google/flan-t5-large `
  --output_dir .\results\experiment_01
```

ก่อนรัน fine stage ตัว loader จะตรวจชื่อ layer และ tensor shape ของ CMMLoc
กับ `my_model` หากนำ checkpoint ผิด architecture มาใส่ จะหยุดพร้อมข้อความ
อธิบายแทนการใช้ `strict=False` แล้วประเมินโมเดลที่โหลดน้ำหนักไม่ครบโดยไม่รู้ตัว
