# Human pose estimation
Human pose estimation is the task of estimating the location of key body joints in an image. This is a very important task in computer vision, as it can be used for many applications, such as human-computer interaction, augmented reality, and even for understanding human behavior.

This repository is based on ultralytics yolov8 model for pose estimation.
It is easy to run, but have a fairly limited set of available models.

For more advanced models, you can check out the [mmpose](https://mmpose.readthedocs.io/en/latest/demos.html) library.

![Human pose estimation](https://github.com/open-mmlab/mmpose/assets/26127467/f14e0129-1e5e-4d74-84fe-28cd62357211)

## Prerequisites

Make sure `uv` is installed and the root environment is synced.

```bash
uv sync
```

## Usage

```bash
uv run main.py
```