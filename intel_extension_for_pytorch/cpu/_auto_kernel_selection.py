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
