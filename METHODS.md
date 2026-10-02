# Methods

## Objective

Reproduce an EEG motor-imagery classification experiment using EEGNet (Lawhern et al., 2018) and the original authors' public implementation.

## Dataset

BCI Competition IV Dataset 2a, subject A01.

- Training session: A01T.gdf
- Evaluation session: A01E.gdf
- 22 EEG + 3 EOG channels
- 250 Hz original sampling rate
- 4 motor-imagery classes
- 288 trials per session

## Preprocessing

1. Keep the first 22 channels from the GDF import.
2. Resample to 128 Hz.
3. Apply a 4–40 Hz band-pass filter.
4. Apply exponential moving standardization.
5. Extract 256-sample trial windows.
6. Format as samples × channels × time × 1 for EEGNet.

For A01T, the window is 0.5–2.5 s after the cue.

For A01E, event 768 marks trial start and the cue occurs 2 s later, so the equivalent window is 2.5–4.5 s after trial start.

## Model

The EEGNet architecture is not reimplemented here. It is cloned and imported directly from:

https://github.com/vlawhern/arl-eegmodels

Configuration:

- nb_classes = 4
- Chans = 22
- Samples = 256
- dropoutRate = 0.5
- kernLength = 32
- F1 = 8
- D = 2
- F2 = 16
- norm_rate = 0.25
- dropoutType = Dropout

Observed parameter count: 1,796.

## Evaluation

A01T is divided into three contiguous 96-trial folds. Each fold uses 192 samples for training and 96 for validation. The best validation checkpoint is then evaluated on the complete A01E session using the official A01E true labels.

## Reproduction caveat

The numerical results in this repository are the user's actual A01 run and are reported as-is. They are not an assertion that all published EEGNet benchmark numbers were reproduced exactly.
