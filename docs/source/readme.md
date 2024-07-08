# SpectralAnalysis
A toolkit for spectral analysis. See [documentation](https://spectralanalysis.readthedocs.io/en) for more details.

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
├── docs/
│   ├── Makefile
│   ├── make.bat
│   └── source/
│       ├── conf.py
│       ├── index.rst
│       └── modules.rst
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
│   │   ├── lssvm.py
│   │   └── svr.py
│   ├── draw.py
│   └── metrics.py
└── main/
    ├── data/
    │   ├── wl_0884.xlsx
    │   ├── wl_1899.xlsx
    │   └── xxxx.xlsx
    ├── config/
    │   └── xxx.yaml      
    ├── result/
    │   └──xxx
    │      └── xxxx
    │          └── YYYYMMDD_HHMM
    │               ├── result.csv
    │               └── xxxxx.png
    ├── README.md
    ├── analyse.py           
    ├── main_controller.py
    └── data_adapter.py  
```

