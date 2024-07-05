# Online Payment Fraud Detection


## Project setup
- Python version: *3.10.6*
- Python version manager: *pyenv* [Installation](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)


#### 1. Virtual environment
- clone the porject
```bash
git clone https://github.com/Weena24/online-payment-fraud-detection.git
cd online-payment-fraud-detection
```
- Create a virtual environement
```bash
pyenv virtualenv fraud-env
```

- Define local python path
```bash
pyenv local fraud-env
```


#### 2. Package installation
```bash
make reinstall_packages
```

#### 3. Raw data initialization
```bash
make initialize_raw_data
```
