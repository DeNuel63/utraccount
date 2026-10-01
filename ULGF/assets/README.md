# ULGF Local Assets

Place the external model and smoke-test files here before uploading the repository to Google Drive.

```text
assets/
├── checkpoint/
│   └── final/
└── smoke/
    ├── images/
    └── labels/
```

- `checkpoint/final/` must contain the extracted Diffusers-format ULGF checkpoint, including `model_index.json` and `generation_config.json`.
- `smoke/images/` should initially contain only one to three test images.
- `smoke/labels/` must contain a YOLO `.txt` label for each image, using the same filename stem.

Example pair:

```text
smoke/images/000001.jpg
smoke/labels/000001.txt
```

Do not commit or publicly redistribute model weights or datasets unless their licenses permit it.
