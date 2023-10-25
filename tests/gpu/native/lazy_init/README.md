# Notice
We should guarantee that only one `TestCase`, aka `test_lazy_init`, exists in `test_lazy_init.py`, which is constantly tested at first.
Otherwise, `pytest` would pollute test environment resulting in a poison fork in multi-processing. 
