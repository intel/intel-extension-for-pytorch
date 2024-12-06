# Run Model inference


1. Command for compilation of example-app 

```
$ cd example-app
$ mkdir build
$ cd build
$ CC=icx CXX=icpx cmake -DCMAKE_PREFIX_PATH=<LIBPYTORCH_PATH> ..
$ make
```

2. Use model_gen.py to generate the resnet50 jit model and save it as resnet50.pt

```
python ../../model_gen.py
```


3. Run example 

```
./example-app resnet50.pt
```
