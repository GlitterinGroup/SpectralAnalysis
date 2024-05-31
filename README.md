# SpectralAnalysis
A toolkit for spectral analysis.

## Install
````python
pip install -r requirements.txt
python setup.py install
````

## Structure
```
SpectralAnalysis/
├── setup.py
├── README.md
├── requirements.txt
├── spectral_analysis/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── feature_selection.py
|   ├── data_split.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── bayes.py
│   │   ├── lr.py
│   │   ├── pls.py
│   │   ├── rf.py
│   │   ├── svm.py
│   │   └── svr.py
│   ├── draw.py
│   └── metrics.py
└── main/
    ├── data/
    │   ├── wl_0884.xlsx
    │   ├── wl_1899.xlsx
    │   └── xxx.xlsx
    ├── config/
    │   └── xxx.yaml      
    ├── result/
    │   └──xxx
    │      └── YYYYMMDD_HHMM
    │           ├── result.csv
    │           └── xxxxxx.png
    ├── README.md
    ├── analyse.py           
    ├── main_controller.py
    └── data_adapter.py  
```

