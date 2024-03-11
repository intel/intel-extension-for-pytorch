from functools import wraps


class WrapAPI:
    @classmethod
    def wrap_api_to(cls, api):
        @wraps(api)
        def new_api(*args, **kwargs):
            device = kwargs.get("device")
            args_len = len(args)
            new_args = list(args)

            if device is not None and str(device).find("cuda") != -1:
                # handle the "cuda:0" to "xpu:0"
                kwargs["device"] = str(device).replace("cuda", "xpu")
            elif device is not None and "device" in kwargs and type(kwargs["device"]) == int:
                # handle the tensor.cuda(device=0) to tensor.to(device='xpu:0')
                kwargs["device"] = "xpu:" + str(kwargs["device"])
            elif device is None and args_len > 1:
                if str(args[1]).find("cuda") != -1:
                    new_device = str(args[1]).replace("cuda", "xpu")
                    new_args[1] = new_device
                elif type(args[1]) == int:
                    new_device = "xpu:" + str(args[1])
                    new_args[1] = new_device
                elif args[1] is None and 'pin_memory' in str(api):
                    # handle the case torch.Tensor.pin_memory(None)
                    new_args[1] = 'xpu'
            elif device is None and args_len == 1:
                kwargs["device"] = "xpu"
            new_args = tuple(new_args)
            return api(*new_args, **kwargs)

        return new_api

    @classmethod
    def wrap_api_common(cls, api):
        @wraps(api)
        def new_api(*args, **kwargs):
            backend = kwargs.get("backend")
            device = kwargs.get("device")
            pin_memory_device = kwargs.get("pin_memory_device")
            args_len = len(args)
            new_args = list(args)
            if backend is not None and backend == "nccl":
                import oneccl_bindings_for_pytorch

                kwargs["backend"] = "ccl"
            elif backend is None and args_len > 0 and args[0] == "nccl":
                import oneccl_bindings_for_pytorch

                new_args[0] = "ccl"

            if device is not None and str(device).find("cuda") != -1:
                kwargs["device"] = str(device).replace("cuda", "xpu")

            if (
                pin_memory_device is not None
                and str(pin_memory_device).find("cuda") != -1
            ):
                kwargs["pin_memory_device"] = str(pin_memory_device).replace(
                    "cuda", "xpu"
                )

            new_args = tuple(new_args)
            return api(*new_args, **kwargs)

        return new_api

    @classmethod
    def wrap_api_skip(cls, api):
        @wraps(api)
        def new_api(*args, **kwargs):
            api_schema = api.__module__ + '.' + api.__name__
            assert False, "Error: the api " + api_schema + " is not supported by xpu"

        return new_api

    @classmethod
    def wrap_api_pass(cls, api):
        @wraps(api)
        def new_api(*args, **kwargs):
            return True

        return new_api

    @classmethod
    def wrap_api_failure(cls, api):
        @wraps(api)
        def new_api(*args, **kwargs):
            print(
                "Warning: This api is not supported by xpu and will automatically return false in any implementation."
            )
            return False

        return new_api

    @classmethod
    def wrap_api_ccl(cls, api):
        @wraps(api)
        def new_api(*args, **kwargs):
            import oneccl_bindings_for_pytorch

            return "ccl"

        return new_api
