# Sources directory

## Project structure

`ann.py` -> Implementation of our Artificial Neural Network

`args.py` -> Arguments parsers

`opso_ann.py` -> Main function for implementation of OPSO (execution file)

`pso.py` -> Implementation of our Particle Swarm Optimization

`pso_ann.py` -> Main function for implementation of PSO (execution file)

`pso_json` -> Load / save pso arguments functions

`train_help` -> Load / save pso arguments functions

`train_help.py` -> Miscellaneous function for ANN & PSO implementation

## PSO Usage 

### Argument description

```
>>> python pso_ann.py -h
usage: PSO to optimize ANN [-h] -f {linear,cubic,tanh,sine,complex,xor}
                           [-pnc PNC] [-b] [-r]

optional arguments:
  -h, --help            show this help message and exit
  -f {linear,cubic,tanh,sine,complex,xor}, --function {linear,cubic,tanh,sine,complex,xor}
                        function to optimize
  -pnc PNC, --pso-number-config PNC
                        number of the config file for pso
  -b                    store config
  -r                    real time graph

```


### Usage Examples

```bash
>>> python pso_ann.py -f linear -r
```
Start pso with `config/linear_pso_0.json` and real time graph

```bash
>>> python pso_ann.py -f cubic -b
```
Start pso with `config/cubic_pso_0.json` and save config

```bash
>>> python pso_ann.py -f tanh -pnc 3
```
Start pso with `config/cubic_tanh_3.json`

## OPSO Usage 

### Argument description

```
>>> python opso_ann.py -h
usage: OPSO training PSO to optimize ANN [-h] -f
                                         {linear,cubic,tanh,sine,complex,xor}
                                         [-onc ONC] [-obc OBC] [-r]

optional arguments:
  -h, --help            show this help message and exit
  -f {linear,cubic,tanh,sine,complex,xor}, --function {linear,cubic,tanh,sine,complex,xor}
                        function to optimize
  -onc ONC, --opso-number-config ONC
                        number of the config file for opso
  -obc OBC, --opso-boundary-config OBC
                        number of the config file for opso
  -r                    real time graph
```

### Usage Examples

```bash
>>> python opso_ann.py -f linear -onc 1 -r
```



