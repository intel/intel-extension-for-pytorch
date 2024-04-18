import logging
import warnings
import functools

format_str = "%(asctime)s - %(filename)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=format_str)

from enum import Enum


class WarningType(Enum):
    NotSupported = 1
    MissingDependency = 2
    MissingArgument = 3
    WrongArgument = 4
    DeprecatedArgument = 5
    AmbiguousArgument = 6


UserFixWarning = {
    WarningType.MissingDependency,
    WarningType.MissingArgument,
    WarningType.WrongArgument,
    WarningType.AmbiguousArgument,
}

WarningType2Prefix = {
    WarningType.NotSupported: "[NotSupported]",
    WarningType.MissingDependency: "[MissingDependency]",
    WarningType.MissingArgument: "[MissingArgument]",
    WarningType.WrongArgument: "[WrongArgument]",
    WarningType.DeprecatedArgument: "[DeprecatedArgument]",
    WarningType.AmbiguousArgument: "[AmbiguousArgument]",
}


class _Logger(logging.Logger):
    """
    An IPEX wrapper for logging.logger
    We use this wrapper for two purpose:
    (1) Unified the usage for warnings.warn and logging.warning: Accroding to
    https://docs.python.org/3/howto/logging.html, we use warnings.warn if the
    issue is avoidable and logging.warn if there is nothing the client
    application can do about the situation.
    (2) Adding more detailed prefixing to the types of the warnings:
     See https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/issues/2618.
     - [NotSupported]
     - [MissingDependency]
     - [MissingArgument]
     - [WrongArgument]
     - [DeprecatedArgument]
     - [AmbiguousArgument]
    """

    def __init__(self, name="IPEX"):
        super(_Logger, self).__init__(name=name)

    def warning(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        warning_t = kwargs.pop("_type", None)
        if warning_t:
            msg = WarningType2Prefix[warning_t] + msg
        if warning_t in UserFixWarning:
            warnings.warn(msg)
        super(_Logger, self).warning(msg, *args, **kwargs)


logging.setLoggerClass(_Logger)
logger = logging.getLogger("IPEX")


def warn_if_user_explicitly_set(user_have_set, msg):
    if user_have_set:
        logger.warning(msg, _type=WarningType.NotSupported)
    else:
        logger.info(msg)


@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    """
    Emit the warning with the same message only once
    """
    self.warning(*args, **kwargs)


logging.Logger.warning_once = warning_once
