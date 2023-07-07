# DLA

## Introduction

Code used to obtain the DLA (Deep Learning Attacker) architecture. The paper was published as Parras, J., Hüttenrauch, M., Zazo, S., & Neumann, G. (2021). Deep Reinforcement Learning for Attacking Wireless Sensor Networks. Sensors, 21(12), 4060. [DOI](https://doi.org/10.3390/s21124060).

## Launch

This project requires Python 3.6. To run this project, create a `virtualenv` (recomended) and then install the requirements as:

```
$ pip install -r requirements.txt
```

To show the results obtained in the paper using the pretrained weights:
```
$ python plot_results.py
```

To train your own weights:

```
$ python run_all_experiments.py
```

Note that training your own weights may take several hours, depending on the configuration of your computer and the number of threads that you set in the script `run_all_experiments.py` (the code uses a single thread by default).

