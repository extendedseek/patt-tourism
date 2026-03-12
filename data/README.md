# Data directory

Expected layout:

```text
data/
├── raw/
│   ├── eurostat.csv or eurostat.json
│   ├── inside_airbnb/
│   │   ├── city_a/
│   │   ├── city_b/
│   │   └── ...
│   └── yelp/
│       ├── yelp_academic_dataset_business.json
│       ├── yelp_academic_dataset_review.json
│       └── photos.json
├── processed/
└── sample/
    ├── MANIFEST.yaml
    ├── city_region_map_sample.csv
    ├── raw/
    └── processed/
```

Notes:
- Processed parquet files created by the scripts are the inputs used by the training configs.
