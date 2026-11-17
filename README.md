# TerraClear

### LISS-IV satellite data analysis and cloud removal 

TerraClear is a Data Mining and Analytics (DMA) project by **Amrit Noor Singh** and **Aryan Gupta**. The project explores how cloud cover affects satellite imagery and how image reconstruction can help make obscured scenes easier to interpret.

We developed the project workflow, including image preparation, patch generation, cloud-mask experiments, model-training experiments, data analysis, evaluation, and a web demo. The satellite imagery is from ISRO's LISS-IV sensor; ISRO is acknowledged as the data source.

## Project links

Add the final links here:

| Resource | Link |
|---|---|
| Hugging Face live demo | `https://huggingface.co/spaces/Amrit2607/liss4-cloud-removal` |
| Kaggle dataset | `https://www.kaggle.com/datasets/amritnoorsingh/isro-data-patches` |
| ISRO imagery source | `https://bhoonidhi.nrsc.gov.in/bhoonidhi/index.html` |


## The problem

LISS-IV is an optical satellite sensor. Clouds can hide land features, making an image less useful for tasks such as observing crops, land cover, or changes after a disaster. The project studies a reconstruction approach: give a model a cloudy image and ask it to estimate the obscured content.

The hidden ground truth cannot be recovered exactly from a cloudy image. A model's output is an estimate, so reconstructions should be treated as visual aids rather than verified replacements for the original satellite observation.

## Data and workflow

The LISS-IV scene in the project contains three separate band files:

| File | Band |
|---|---|
| `BAND2.tif` | Green |
| `BAND3.tif` | Red |
| `BAND4.tif` | Near-infrared (NIR) |

The workflow is:

1. Prepare the separate LISS-IV bands and generate image patches.
2. Create cloud masks and cloudy examples for reconstruction experiments.
3. Train and evaluate an image-reconstruction model using image patches.
4. Analyze the patches with descriptive statistics, texture features, frequency analysis, clustering, spectral indices, cloud/clear comparisons, and quality scoring.
5. Run inference through a Streamlit app using an ONNX model.

`lissiv_pipeline.py` combines Red, Green, and NIR into a false-colour image and saves overlapping PNG patches. It stretches each band independently for display, so those PNG pixel values should not be treated as calibrated reflectance when calculating scientific spectral indices. The original band rasters are needed for reflectance-based analysis.

## Demo and evaluation

The web app has two modes:

- **Upload an image:** upload a supported image or array and inspect the reconstruction. For real uploads, there is no ground truth, so the result is qualitative.
- **Random demo patch:** choose a clear patch, add a synthetic cloud, reconstruct it, and compare the result with the known clear patch using PSNR and SSIM.

The saved demo figure in `checker_outputs/onnx_demo_grid.png` reports **average SSIM 0.7689** and **average PSNR 19.19 dB** across 8 demo patches. These are results from a small synthetic-cloud demo, not a measure of performance on real clouds across the full dataset.

## Repository contents

```text
.
├── app.py                    # Streamlit inference and demo app
├── model.onnx                # ONNX inference model used by the app
├── lissiv_pipeline.py        # Prepare patches from separate LISS-IV bands
├── data_pipeline.py          # Prepare patches from multiband TIFF files
├── clean_bucket.py           # Remove PNG patches with large black regions
├── isro_analysis.ipynb       # Data-mining and image-analysis notebook
├── model_train.ipynb         # Model-training and ONNX export experiments
├── checker_outputs/          # Saved model-checking visualizations
├── Dockerfile                # Container setup for the Streamlit app
├── requirements.txt          # Inference/deployment dependencies
├── .gitattributes            # Git LFS rules for large model files
├── .gitignore
└── .dockerignore
```

`lissiv_data/` and `demo_patches/` are local-only folders and are intentionally excluded from Git. Download or prepare those files using the Kaggle or ISRO links above. The ONNX model is configured for Git LFS; install Git LFS and run `git lfs install` before committing it.

## Run the web app locally

The app expects `model.onnx` in the project folder. Demo mode also needs at least one suitable 4-channel `.npy` patch in `demo_patches/`.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Run with Docker

```bash
docker build -t terraclear .
docker run --rm -p 7860:7860 terraclear
```

Then open `http://localhost:7860`.

The Docker build excludes local imagery and demo patches. Image uploads still work, but random demo-patch mode requires suitable `.npy` patches to be made available inside the container.

## Prepare LISS-IV patches

The provided scene is stored under `lissiv_data/`. `lissiv_pipeline.py` can process scenes with separate `BAND2.tif`, `BAND3.tif`, and `BAND4.tif` files. Its default input path is `./data/lissiv_data`; either place the scene there or update `ROOT_DIR` to `./lissiv_data` before running it.

```bash
python lissiv_pipeline.py
```

The cleanup script deletes matching PNG files, so check `PATCH_FOLDER` before running it:

```bash
python clean_bucket.py
```

## Training and model interface

`model_train.ipynb` contains training and export experiments. The notebook currently defines a 3-channel model, while `app.py` calls ONNX with a 4-channel image and a separate cloud-mask input. These interfaces need to match to reproduce the exact model used by the app. The deployed `model.onnx` is the inference artifact; confirm its source and input/output contract before replacing it with a newly exported model.

The training notebook uses Kaggle paths. Update its dataset paths to match the Kaggle dataset before running it. The Docker requirements are for inference and do not install PyTorch for training.

## Data and usage notes

- The Kaggle dataset and the source imagery should be cited using the links above.
- Check the source-data terms before redistributing raw ISRO imagery or derived patches.
- Synthetic-cloud demo results do not guarantee performance on real cloud formations.
- The app's output is an estimate and should not be used as authoritative scientific data without validation.

## Team

- **Amrit Noor Singh(24117003)**
- **Aryan Gupta(24117005)**


