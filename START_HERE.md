# Start Here

## 1. Open Colab

Create a fresh Google Colab notebook and select **CPU** runtime.

## 2. Install dependencies

```bash
!pip -q install tensorflow==2.20.0 mne scipy pandas matplotlib
```

## 3. Clone the original EEGNet repository

```bash
!git clone -q https://github.com/vlawhern/arl-eegmodels.git /content/arl-eegmodels
```

## 4. Download data

Official dataset page:
https://www.bbci.de/competition/iv/download/

Dataset archive:
https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip

Evaluation labels:
https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip

## 5. Run

Use `reproduce_eegnet.py` or the included notebook.

## 6. Expected reference result

The documented A01 run produced:

- 1,796 parameters
- 69.56% mean E-session accuracy
- 0.16 percentage-point standard deviation
- 28.61 minutes total CPU training time
