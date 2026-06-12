# Data and embedding profiles

Raw medical images, checkpoints, and full embedding matrices are intentionally
not stored in this repository.

The paper uses five brain-MRI datasets and nine CNN backbones, giving 45
primary dataset-backbone contexts. HAM10000 is an external-domain control and
is excluded from the main brain-MRI averages and inferential tests.

Known public dataset pages used by the project include:

- Brain tumor MRI 4C:
  <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>
- Brain tumor MRI 17C:
  <https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes>
- Brain tumor MRI 44C:
  <https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c>

The 15C, ScienceDB 3C, and HAM10000 copies must be obtained from their original
providers according to their licenses. See the paper bibliography and dataset
audit before redistributing any data.

## Expected embedding profile

```text
profile/
  split_info.json
  embeddings/
    multicapa_norm/
      train/
        labels.npy
        z_dim_64.npy
        z_dim_128.npy
        ...
      val/
        labels.npy
        z_dim_64.npy
        ...
      test/
        labels.npy
        z_dim_64.npy
        ...
```

Each `z_dim_*.npy` file is a two-dimensional matrix with one row per sample.
All views within a split must have the same row order and labels.

Run:

```bash
python data/prepare_data.py --profile /path/to/profile
```

before executing HMDF-kNN.
