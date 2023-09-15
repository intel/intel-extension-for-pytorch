_use_dnnl = False


def _enable_dnnl():
    global _use_dnnl
    _use_dnnl = True


def _disable_dnnl():
    global _use_dnnl
    _use_dnnl = False


def _using_dnnl():
    global _use_dnnl
    return _use_dnnl


_use_tpp = False


def _enable_tpp():
    global _use_tpp
    _use_tpp = True


def _disable_tpp():
    global _use_tpp
    _use_tpp = False


def _using_tpp():
    global _use_tpp
    return _use_tpp
