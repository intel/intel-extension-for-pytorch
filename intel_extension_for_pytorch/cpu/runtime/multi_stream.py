import torch
import torch.nn as nn
from typing import Union
import intel_extension_for_pytorch._C as core
from .cpupool import CPUPool
from .task import Task
import copy
from ...utils._logger import logger, WarningType


class MultiStreamModuleHint(object):
    r"""
    MultiStreamModuleHint is a hint to MultiStreamModule about how to split the inputs
    or concat the output. Each argument should be None, with type of int or a container
    which containes int or None such as: (0, None, ...) or [0, None, ...]. If the argument
    is None, it means this argument will not be split or concat. If the argument is with
    type int, its value means along which dim this argument will be split or concat.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        intel_extension_for_pytorch.cpu.runtime.MultiStreamModuleHint: Generated
        intel_extension_for_pytorch.cpu.runtime.MultiStreamModuleHint object.

    :meta public:
    """

    def __init__(self, *args, **kwargs):
        self.args = list(args)
        self.kwargs = kwargs
        self.args_len = args.__len__()
        self.kwargs_len = kwargs.__len__()


default_multi_stream_module_split_hint = MultiStreamModuleHint(0)
default_multi_stream_module_concat_hint = MultiStreamModuleHint(0)


def get_default_num_streams(cpu_pool):
    # One core per stream usually brings better overall throughput than other configurations.
    # Therefore, we heuristically make one core per stream the default here.
    return cpu_pool.core_ids.__len__()


class MultiStreamModule(nn.Module):
    r"""
    MultiStreamModule supports inference with multi-stream throughput mode.

    If the number of cores inside ``cpu_pool`` is divisible by ``num_streams``,
    the cores will be allocated equally to each stream. If the number of cores
    inside ``cpu_pool`` is not divisible by ``num_streams`` with remainder N,
    one extra core will be allocated to the first N streams. We suggest to set
    the ``num_streams`` as divisor of core number inside ``cpu_pool``.

    If the inputs' batchsize is larger than and divisible by ``num_streams``,
    the batchsize will be allocated equally to each stream. If batchsize is not
    divisible by ``num_streams`` with remainder N, one extra piece will be
    allocated to the first N streams. If the inputs' batchsize is less than
    ``num_streams``, only the first batchsize's streams are used with mini batch
    as one. We suggest to set inputs' batchsize larger than and divisible by
    ``num_streams``. If you don't want to tune the num of streams and leave it
    as "AUTO", we suggest to set inputs' batchsize larger than and divisible by
    number of cores.

    Args:
        model (torch.jit.ScriptModule or torch.nn.Module): The input model.
        num_streams (Union[int, str]): Number of instances (int) or "AUTO" (str). "AUTO" means the stream number
            will be selected automatically. Although "AUTO" usually provides a
            reasonable performance, it may still not be optimal for some cases which
            means manual tuning for number of streams is needed for this case.
        cpu_pool (intel_extension_for_pytorch.cpu.runtime.CPUPool): An
            intel_extension_for_pytorch.cpu.runtime.CPUPool object, contains
            all CPU cores used to run multi-stream inference.
        concat_output (bool): A flag indicates whether the output of each
            stream will be concatenated or not. The default value is True. Note:
            if the output of each stream can't be concatenated, set this flag to
            false to get the raw output (a list of each stream's output).
        input_split_hint (MultiStreamModuleHint): Hint to MultiStreamModule about
            how to split the inputs.
        output_concat_hint (MultiStreamModuleHint): Hint to MultiStreamModule about
            how to concat the outputs.

    Returns:
        intel_extension_for_pytorch.cpu.runtime.MultiStreamModule: Generated
        intel_extension_for_pytorch.cpu.runtime.MultiStreamModule object.

    :meta public:
    """

    def __init__(
        self,
        model,
        num_streams: Union[int, str] = "AUTO",
        cpu_pool: CPUPool = CPUPool(),
        concat_output: bool = True,
        input_split_hint: MultiStreamModuleHint = default_multi_stream_module_split_hint,
        output_concat_hint: MultiStreamModuleHint = default_multi_stream_module_concat_hint,
    ):
        super(MultiStreamModule, self).__init__()
        assert (
            type(cpu_pool) is CPUPool
        ), "Input of cpu_pool must be provided with type of ipex.cpu.runtime.CPUPool"
        if not isinstance(model, torch.jit.ScriptModule):
            logger.warning(
                "Creating MultiStreamModule on an nn.Module. This can be slow due "
                + "to Python Global Interpreter Lock (GIL). Suggest to use JIT ScriptModule for better performance.",
                _type=WarningType.WrongArgument,
            )
        self.cpu_pool = cpu_pool
        self.core_list = cpu_pool.core_ids
        if isinstance(num_streams, str):
            # For str input of num_streams, it must be "auto"
            if num_streams.upper() == "AUTO":
                self.num_streams = get_default_num_streams(
                    cpu_pool
                )  # The default selected value when auto selection is on.
            else:
                AssertionError(
                    False
                ), 'Input of num_streams must be Number of instances or string "AUTO"'
        else:
            assert isinstance(
                num_streams, int
            ), 'Input of num_streams must be Number of instances or string "auto"'
            self.num_streams = num_streams

        if self.num_streams > self.core_list.__len__():
            self.num_streams = self.core_list.__len__()
            logger.warning(
                f"The number of streams is larger than number of cores. The number of streams changes to {self.num_streams}.",
                _type=WarningType.WrongArgument,
            )

        if self.num_streams == 1:
            # Sync execution path if num_stream is 1.
            self.model = model
        else:
            self.cores_per_instance = self.core_list.__len__() // self.num_streams
            num_stream_allocated_extra_core = (
                self.core_list.__len__() % self.num_streams
            )
            self.tasks = []
            start_core_list_idx = 0
            end_core_list_idx = 0
            for j in range(self.num_streams):
                if j < num_stream_allocated_extra_core:
                    # If the core number is not divisible by stream number,
                    # the remainder streams(num_stream_allocated_extra_core) will be allocated one extra core.
                    end_core_list_idx += self.cores_per_instance + 1
                else:
                    end_core_list_idx += self.cores_per_instance
                self.tasks.append(
                    Task(
                        model,
                        CPUPool(self.core_list[start_core_list_idx:end_core_list_idx]),
                    )
                )
                start_core_list_idx = end_core_list_idx
        self.concat_output = concat_output
        self.input_split_hint = input_split_hint
        self.output_concat_hint = output_concat_hint

        # Deep copy the input structure for each stream based on input_split_hint.
        # Each streams_input will be recursively visited and set to the split value in place.
        self.args_streams_input = []
        self.kwargs_streams_input = []
        for _ in range(self.num_streams):
            self.args_streams_input.append(copy.deepcopy(self.input_split_hint.args))
            self.kwargs_streams_input.append(
                copy.deepcopy(self.input_split_hint.kwargs)
            )

        # Deep copy the output structure based on output_concat_hint.
        # self.output will be recursively visited and set to the concat value in place.
        self.output = copy.deepcopy(self.output_concat_hint)

        # Init status needed for forward
        self.reset_forward_status()

    def reset_forward_status(self):
        # Since the input batchsize for each forward invoking may change
        # Need to reset the status for each forward invoking
        #   * split_size: will be calculated by input batch size and num_streams.
        #   * used_num_streams: is the num_streams actually used by this forward invoking.
        #       It may less than self.num_streams when bs is less than self.num_streams.
        #   * current_split_start_idx: used to record the split start idx for current stream.
        #   * current_split_end_idx: used to record the split end idx for current stream.
        self.split_size = None
        self.used_num_streams = self.num_streams
        self.current_split_start_idx = 0
        self.current_split_end_idx = 0

    def update_split_idx(self, stream_id):
        # Set current_split_start_idx to last current_split_end_idx
        self.current_split_start_idx = self.current_split_end_idx
        # Calculate current_split_end_idx to new value
        if stream_id < self.instance_need_extra_input:
            # Tail case, when the input image size larger than num_streams and not divisible,
            # the first remainder streams will have (mini_batch + 1) input size.
            self.current_split_end_idx = self.current_split_end_idx + (
                self.batch_per_instance + 1
            )
        else:
            # Input image size divisible of num_streams or input image size less than num_streams.
            self.current_split_end_idx = (
                self.current_split_end_idx + self.batch_per_instance
            )

    def init_forward_status(self, split_size, stream_id):
        # This function should be invoke only once at each forward
        self.split_size = split_size
        # Ensure each instance has input offload
        self.batch_per_instance = self.split_size // self.num_streams
        if self.batch_per_instance >= 1:
            # The input batchsize larger or equal to num_streams.
            self.used_num_streams = self.num_streams
            # If input batchsize larger than num_streams and not divisible,
            # the first remainder streams will have (mini_batch + 1) input size.
            self.instance_need_extra_input = self.split_size % self.num_streams
        else:
            # The input batchsize less than num_streams,
            # only the first batchsize stream will have mini_batch(1) input.
            self.batch_per_instance = 1
            self.used_num_streams = self.split_size
            self.instance_need_extra_input = 0
        self.update_split_idx(stream_id)

    def _do_get_input_for_each_stream(
        self, hint_object, input_object, stream_input_object, idx_or_key, stream_id
    ):
        # * hint_object: input hint to tell whether we need to split corresponding
        #       input_object at current position.
        # * input_object: raw input used to split and generate stream_input_object
        #       at current position.
        # * stream_input_object: Which will generated in place as the input for
        #       current stream (marked by stream_id)
        # * idx_or_key: idx (for list/tuple) and key (for dict) used for recursive
        #       visit of hint_object/input_object/stream_input_object.
        # * stream_id: the stream we are visiting now.
        type_arg = type(hint_object[idx_or_key])
        if type_arg in [list]:
            for i in range(hint_object[idx_or_key].__len__()):
                self._do_get_input_for_each_stream(
                    hint_object[idx_or_key],
                    input_object[idx_or_key],
                    stream_input_object[idx_or_key],
                    i,
                    stream_id,
                )
        if type_arg in [tuple]:
            # Tuple doesn't support item change in place
            # So we change it to list for next recursion and change it back to tuple.
            temp = list(stream_input_object[idx_or_key])
            for i in range(hint_object[idx_or_key].__len__()):
                self._do_get_input_for_each_stream(
                    hint_object[idx_or_key],
                    input_object[idx_or_key],
                    temp,
                    i,
                    stream_id,
                )
            stream_input_object[idx_or_key] = tuple(temp)
        elif type_arg in [dict]:
            for key in hint_object[idx_or_key]:
                self._do_get_input_for_each_stream(
                    hint_object[idx_or_key],
                    input_object[idx_or_key],
                    stream_input_object[idx_or_key],
                    key,
                    stream_id,
                )
        elif (type_arg is int) or (hint_object[idx_or_key] is None):
            if hint_object[idx_or_key] is not None:
                # If user tells us to split in this object,
                if self.split_size is None:
                    # Init the input status for each stream here
                    # Here the stream_id must be 0
                    self.init_forward_status(
                        input_object[idx_or_key].size(hint_object[idx_or_key]),
                        stream_id,
                    )
                # Get the split input for each stream
                # Here we assume split along the outside dim, otherwise memory copy happens and obviously \
                # hurt multi stream module's performance.
                if hint_object[idx_or_key] == 0:
                    # Split along dim 0, the slice will not create new tensor
                    stream_input_object[idx_or_key] = input_object[idx_or_key][
                        self.current_split_start_idx : self.current_split_end_idx
                    ]
                else:
                    # Otherwise, we use torch.narrow
                    length = self.current_split_end_idx - self.current_split_start_idx
                    stream_input_object[idx_or_key] = input_object[idx_or_key].narrow(
                        hint_object[idx_or_key], self.current_split_start_idx, length
                    )
            else:
                # This object shouldn't be split, just set it as each stream's input
                stream_input_object[idx_or_key] = input_object[idx_or_key]
        else:
            AssertionError(
                False
            ), "Generate stream input failed, unsupport input hint type of:{}".format(
                type_arg
            )
        return None

    def _get_input_for_each_stream(
        self, multi_stream_module_split_hint, *args, **kwargs
    ):
        # recursive once to init:
        #   1. Decide the actual self.used_num_streams (it may less than number stream when input bs is small)
        #   2. Init the current_split_start_idx and current_split_end_idx for inputs split
        #   3. Decide the actual input for stream_id 0
        for i in range(multi_stream_module_split_hint.args_len):
            self._do_get_input_for_each_stream(
                hint_object=multi_stream_module_split_hint.args,
                input_object=args,
                stream_input_object=self.args_streams_input[0],
                idx_or_key=i,
                stream_id=0,
            )
        for key in multi_stream_module_split_hint.kwargs:
            self._do_get_input_for_each_stream(
                hint_object=multi_stream_module_split_hint.kwargs,
                input_object=kwargs,
                stream_input_object=self.kwargs_streams_input[0],
                idx_or_key=key,
                stream_id=0,
            )
        # After we get the self.used_num_streams then we can
        # decide the inputs for the left of used_num_streams
        for stream_id in range(1, self.used_num_streams):
            # Update the split idx for current stream
            self.update_split_idx(stream_id)
            # Here we put stream go through as the outer for loop,
            # Since we assume the multi_stream_module_split_hint is not complicated to be recursive generally.
            for i in range(multi_stream_module_split_hint.args_len):
                self._do_get_input_for_each_stream(
                    hint_object=multi_stream_module_split_hint.args,
                    input_object=args,
                    stream_input_object=self.args_streams_input[stream_id],
                    idx_or_key=i,
                    stream_id=stream_id,
                )
            for key in multi_stream_module_split_hint.kwargs:
                self._do_get_input_for_each_stream(
                    hint_object=multi_stream_module_split_hint.kwargs,
                    input_object=kwargs,
                    stream_input_object=self.kwargs_streams_input[stream_id],
                    idx_or_key=key,
                    stream_id=stream_id,
                )

    def _do_generate_outputs(
        self, hint_object, output_object, stream_output_object, idx_or_key, stream_id
    ):
        type_arg = type(hint_object[idx_or_key])
        if type_arg in [list]:
            for i in range(hint_object[idx_or_key].__len__()):
                self._do_generate_outputs(
                    hint_object[idx_or_key],
                    output_object[idx_or_key],
                    stream_output_object[idx_or_key],
                    i,
                    stream_id,
                )
        elif type_arg in [tuple]:
            # Tuple doesn't support item change in place
            # So we change it to list for next recursion and change it back to tuple.
            temp = list(output_object[idx_or_key])
            for i in range(hint_object[idx_or_key].__len__()):
                self._do_generate_outputs(
                    hint_object[idx_or_key],
                    temp,
                    stream_output_object[idx_or_key],
                    i,
                    stream_id,
                )
            output_object[idx_or_key] = tuple(temp)
        elif type_arg in [dict]:
            for key in hint_object[idx_or_key]:
                self._do_generate_outputs(
                    hint_object[idx_or_key],
                    output_object[idx_or_key],
                    stream_output_object[idx_or_key],
                    key,
                    stream_id,
                )
        elif (type_arg is int) or (hint_object[idx_or_key] is None):
            if hint_object[idx_or_key] is not None:
                if stream_id == 0:
                    output_object[idx_or_key] = []
                output_object[idx_or_key].append(stream_output_object[idx_or_key])
            else:
                # This object shouldn't be concat, just copy once for stream_id = 0
                if stream_id == 0:
                    output_object[idx_or_key] = stream_output_object[idx_or_key]
        else:
            AssertionError(
                False
            ), "Generate outputs failed, unsupport output hint type of:{}".format(
                type_arg
            )
        return None

    def _generate_outputs(self, stream_output_object, stream_id):
        # For each position, we will push the result generated by each stream into the list
        # multi_stream_module_split_hint.args_len must be 1, since the module output will be a \
        # single output or tuple for multi outputs
        if self.output_concat_hint.args:
            self._do_generate_outputs(
                hint_object=self.output_concat_hint.args,
                output_object=self.output.args,
                stream_output_object=stream_output_object,
                idx_or_key=0,
                stream_id=stream_id,
            )
        if self.output_concat_hint.kwargs:
            for key, value in self.output_concat_hint.kwargs.items():
                self._do_generate_outputs(
                    hint_object=self.output_concat_hint.kwargs,
                    output_object=self.output.kwargs,
                    stream_output_object=stream_output_object[0],
                    idx_or_key=key,
                    stream_id=stream_id,
                )

    def _do_concat_output_for_each_stream(self, hint_object, output_object, idx_or_key):
        type_arg = type(hint_object[idx_or_key])
        if type_arg in [list]:
            for i in range(hint_object[idx_or_key].__len__()):
                self._do_concat_output_for_each_stream(
                    hint_object[idx_or_key], output_object[idx_or_key], i
                )
        if type_arg in [tuple]:
            # Tuple doesn't support item change in place
            # So we change it to list for next recursion and change it back to tuple.
            temp = list(output_object[idx_or_key])
            for i in range(hint_object[idx_or_key].__len__()):
                self._do_concat_output_for_each_stream(hint_object[idx_or_key], temp, i)
            output_object[idx_or_key] = tuple(temp)
        elif type_arg in [dict]:
            for key in hint_object[idx_or_key]:
                self._do_concat_output_for_each_stream(
                    hint_object[idx_or_key], output_object[idx_or_key], key
                )
        elif (type_arg is int) or (hint_object[idx_or_key] is None):
            if hint_object[idx_or_key] is not None:
                output_object[idx_or_key] = torch.cat(
                    output_object[idx_or_key], dim=hint_object[idx_or_key]
                )
        else:
            AssertionError(
                False
            ), "Concat output failed, unsupport output hint type of:{}".format(type_arg)
        return None

    def _concat_output_for_each_stream(self):
        # Concat the output, when here each position is already a List of tensors to be concat.
        if self.output_concat_hint.args:
            self._do_concat_output_for_each_stream(
                self.output_concat_hint.args, self.output.args, 0
            )

        return_obj = dict()
        if self.output_concat_hint.kwargs:
            for key, value in self.output_concat_hint.kwargs.items():
                self._do_concat_output_for_each_stream(
                    self.output_concat_hint.kwargs, self.output.kwargs, key
                )
                return_obj[key] = self.output.kwargs[key]

        # If the output hint has both the args and kwargs, then we return them as a tuple.
        # Otherwise, return them as it is.
        if self.output_concat_hint.args and self.output_concat_hint.kwargs:
            return self.output.args[0], return_obj
        elif self.output_concat_hint.args:
            return self.output.args[0]
        else:
            return return_obj

    def forward(self, *args, **kwargs):
        # Reset the forward status to default value which mainly contains information
        # to split inputs. They will init afterwards for each forward call.
        self.reset_forward_status()
        if self.num_streams == 1:
            # Sync execution path if num_stream is 1
            if not core.is_same_core_affinity_setting(self.core_list):
                # If the main thread's core affinity has been changed, we should set it again.
                core.pin_cpu_cores(self.cpu_pool.cpu_pool)
            results_raw = self.model(*args, **kwargs)
            return results_raw if self.concat_output else [results_raw]

        # Split the raw input to generate input for each stream
        self._get_input_for_each_stream(self.input_split_hint, *args, **kwargs)

        results_raw_future = []
        results_raw = []
        for stream_id in range(self.used_num_streams):
            results_raw_future.append(
                self.tasks[stream_id](
                    *(self.args_streams_input[stream_id]),
                    **(self.kwargs_streams_input[stream_id]),
                )
            )

        for stream_id in range(self.used_num_streams):
            # If we need to concat the output, for each position, we will push the result generated \
            # by each stream into a list for concat later.
            # For self._generate_outputs: here we put results_raw_future[stream_id].get() into a \
            # [results_raw_future[stream_id].get()]
            # to align the multi_stream_module_concat_hint structure.
            (
                self._generate_outputs([results_raw_future[stream_id].get()], stream_id)
                if self.concat_output
                else results_raw.append(results_raw_future[stream_id].get())
            )
        # If we need to concat the output, for each position, we will concat the result in the list \
        # (generate in self._generate_outputs).
        return (
            self._concat_output_for_each_stream() if self.concat_output else results_raw
        )

    def get_stream_number(self):
        return self.num_streams


class _MultiStreamBenchmarkModule(nn.Module):
    # Here is an internal Module for weight sharing benchmark
    # The diffence with MultiStreamModule:
    #    * The input will not be split. So each stream will run with the same input.
    #    * The output will not be concat. But synchronization point for each stream still exsits at the end \
    #      of the forward method.
    def __init__(
        self,
        model,
        num_streams: Union[int, str] = "AUTO",
        cpu_pool: CPUPool = CPUPool(),
    ):
        super(_MultiStreamBenchmarkModule, self).__init__()
        assert (
            type(cpu_pool) is CPUPool
        ), "Input of cpu_pool must be provided with type of ipex.cpu.runtime.CPUPool"
        self.cpu_pool = cpu_pool
        self.core_list = cpu_pool.core_ids
        if isinstance(num_streams, str):
            # For str input of num_streams, it must be "auto"
            if num_streams.upper() == "AUTO":
                self.num_streams = get_default_num_streams(
                    cpu_pool
                )  # The default selected value when auto selection is on.
            else:
                AssertionError(
                    False
                ), 'Input of num_streams must be Number of instances or string "AUTO"'
        else:
            assert isinstance(
                num_streams, int
            ), 'Input of num_streams must be Number of instances or string "auto"'
            self.num_streams = num_streams

        if self.num_streams > self.core_list.__len__():
            self.num_streams = self.core_list.__len__()
            logger.warning(
                f"The number of streams is larger than number of cores. The number of streams changes to {self.num_streams}.",
                _type=WarningType.WrongArgument,
            )

        if self.num_streams == 1:
            # Sync execution path if num_stream is 1.
            self.model = model
        else:
            self.cores_per_instance = self.core_list.__len__() // self.num_streams
            num_stream_allocated_extra_core = (
                self.core_list.__len__() % self.num_streams
            )
            self.tasks = []
            start_core_list_idx = 0
            end_core_list_idx = 0
            for j in range(self.num_streams):
                if j < num_stream_allocated_extra_core:
                    # If the core number is not divisible by stream number,
                    # the remainder streams(num_stream_allocated_extra_core) will be allocated one extra core.
                    end_core_list_idx += self.cores_per_instance + 1
                else:
                    end_core_list_idx += self.cores_per_instance
                self.tasks.append(
                    Task(
                        model,
                        CPUPool(self.core_list[start_core_list_idx:end_core_list_idx]),
                    )
                )
                start_core_list_idx = end_core_list_idx

    def forward(self, *args, **kwargs):
        if self.num_streams == 1:
            # Sync execution path if num_stream is 1
            if not core.is_same_core_affinity_setting(self.core_list):
                # If the main thread's core affinity has been changed, we should set it again.
                core.pin_cpu_cores(self.cpu_pool.cpu_pool)
            return self.model(*args, **kwargs)

        results_raw_future = []
        results_raw = []
        for j in range(self.num_streams):
            results_raw_future.append(self.tasks[j](*args, **kwargs))

        for j in range(self.num_streams):
            results_raw.append(results_raw_future[j].get())
        return results_raw[0]
