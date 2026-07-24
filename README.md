# CMMLoc coarse + CMMLoc/MNCL/CMMLoc_MNCLv4 fine

Pipeline นี้ใช้ **CMMLoc coarse เพียงครั้งเดียว** เพื่อสร้าง top-k cells
ชุดเดียวกัน แล้วส่งให้ fine stage สาม backend:

- `CMMLoc` — official CMMLoc fine architecture + standard T5-large
- `MNCL` — official MNCL architecture ใน subprocess แยก + Flan-T5-large
- `my_model` — fine architecture จากงาน `CMMLoc_MNCLv4` + standard T5-large

ไฟล์ `.pth` ไม่ได้ใช้ข้าม architecture แม้บาง checkpoint จะมีชื่อ tensor
เหมือนกันก็ตาม แต่ละโมเดลถูกสร้างเป็น instance ใหม่ โหลดแบบตรวจชื่อและ shape
ของ active tensors และ reset seed ก่อน fine inference เพื่อให้ point sampling
เหมือนกันทุก backend

## โครงสร้างไฟล์

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
└── third_party/
    └── MNCL/
```

`data/`, `checkpoints/`, `third_party/MNCL/`, model cache และ `results/`
ไม่ถูก commit เพราะมีขนาดใหญ่หรือสร้างใหม่ได้

## เตรียมเครื่อง GPU

หลัง `git pull` ให้รันคำสั่ง setup แยกจากคำสั่ง Python:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_mncl.ps1
```

script จะติดตั้ง official MNCL ที่ revision
`11ea10e1658b38e53b2127f4ee55f9d4236d9f50` ซึ่งเป็น revision ที่ worker
ตรวจสอบก่อนรัน หากเคย clone MNCL ไว้แล้ว script จะตรวจ revision และ tracked
changes แทนการเขียนทับ

Backbone ที่ต้องใช้:

- CMMLoc: `google-t5/t5-large`
- MNCL: `google/flan-t5-large`
- `my_model` (`CMMLoc_MNCLv4`): `google-t5/t5-large`

ห้ามใช้ Flan-T5 กับ CMMLoc หรือ `my_model` เพราะ checkpoint สามารถโหลดได้โดย
ไม่เกิด shape error แต่ text embeddings จะไม่ตรงกับตอน train

## ตรวจ coarse ก่อน

```powershell
python -m evaluation.hybrid_pipeline `
  --cmmloc_t5_path google-t5/t5-large `
  --use_test_set `
  --coarse_only `
  --top_k 1 3 5 10 `
  --threshs 5 10 15
```

checkpoint CMMLoc ที่ตรวจแล้วควรได้ exact-cell retrieval บน test set ประมาณ:

```text
top-1  0.3233
top-3  0.5343
top-5  0.6313
top-10 0.7376
```

ค่าที่ได้ล่าสุดตรงกับ official CMMLoc; ดังนั้นหากค่ารอบใหม่ต่างมากให้หยุดก่อน
fine stage และตรวจ checkpoint, T5 backbone, dataset และ split

pipeline จะหยุดทันทีหาก SHA-256 ของ checkpoint ไม่ตรงกับชุดที่ audit:

```text
CMMLoc/coarse.pth  5e14e158c3de1fc046d9b970ef1d06c6d4a98d55a1cfdd09f6d26dfc23076f85
CMMLoc/fine.pth    720623e7e25866b0e552b83080202bd3ec855672e0bc83cd962c173506cd648a
MNCL/fine.pth      0a1727faf5108518a83ec182ced6b0e6594f8190267c1533c22c525ea0c62dd4
my_model/fine.pth  acea2f8fbe58aae256d942606dfd269fa9c3b486849b64edd24cf8720c1fbe1e
```

## รันครบสาม fine models

```powershell
python -m evaluation.hybrid_pipeline `
  --cmmloc_t5_path google-t5/t5-large `
  --mncl_t5_path google/flan-t5-large `
  --my_model_t5_path google-t5/t5-large `
  --my_model_text_max_length 128 `
  --eval_seed 42 `
  --use_test_set `
  --top_k 1 3 5 10 `
  --threshs 5 10 15
```

ค่า `128` ตรงกับ default ของ `CMMLoc_MNCLv4/evaluation/args.py` และ evaluation
scripts ของงาน หากเปลี่ยนไปใช้ checkpoint ที่ train ด้วย token limit อื่น
ต้องระบุค่าที่ตรงกับการ train ผ่าน `--my_model_text_max_length`

เลือกทดสอบเพียงบาง backend ได้ เช่น:

```powershell
python -m evaluation.hybrid_pipeline `
  --cmmloc_t5_path google-t5/t5-large `
  --fine_models CMMLoc `
  --use_test_set `
  --top_k 1 3 5 10 `
  --threshs 5 10 15
```

## ผลลัพธ์

```text
results/cmmloc_coarse_multi_fine/
├── cmmloc_retrievals.json
├── cmmloc_fine.json
├── mncl_fine.json
├── my_model_fine.json
└── comparison.json
```

แต่ละไฟล์บันทึก backend, text backbone, checkpoint path/SHA-256, load report,
seed และ accuracy จึงตรวจย้อนหลังได้ว่าใช้ artifact ใดจริง และผลของแต่ละ
fine model จะถูกเขียนทันทีเมื่อ model นั้นจบ ไม่สูญหายหากขั้นถัดไปหยุด

## การตีความผลอย่างถูกต้อง

นี่คือการเปรียบเทียบ **fine stage ภายใต้ CMMLoc retrievals ชุดเดียวกัน**
จึงเป็นการวัดว่า fine backend ใดทำงานได้ดีกว่าบน candidate cells เดียวกัน
ผล MNCL และ `my_model` ไม่จำเป็นต้องเท่ากับ end-to-end result เดิมของแต่ละงาน
เพราะ original pipelines อาจใช้ coarse model ของตัวเอง

สำหรับ CMMLoc ซึ่งใช้ coarse และ fine ของ official model เหมือนเดิม ค่า test
fine localization ใน paper อยู่ประมาณ:

```text
        5 m   10 m  15 m
top-1   0.39  0.53  0.56
top-3   0.67  0.80  0.82
top-5   0.77  0.87  0.89
```

ค่าจริงอาจต่างเล็กน้อยจาก point sampling/library version แต่ไม่ควรตกลงใกล้ศูนย์
เหมือนกรณีใช้ text backbone หรือ backend ผิดรุ่น
