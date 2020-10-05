test_lazy_reorder

We assum dpcpp op as 1, not support op is 0.You can design the pattern like sequential pattern such as 000,111,110,101,011,100,010,001.
And you also can design a shortcut structure like add(conv,conv).
We will collect more pattern from public model to vreify laze_reorder function later.
You can use like this to generate mkldnn log:
```
python test_lazy_reorder_design.py > mkldnn.log

```
Then,you can parse the mkldnn log like this:

```
python test_lazy_reorder_parser.py --file mkldnn.log

```
You will get calls of reorder like below:

```
primtive                time (ms)               calls
reorder                 0.2561035               3
convolution             0.0859375               2
sum                     0.052002                1

total time: 0.394043 ms for 3 items.

```
We also can get more information from the log that we need.