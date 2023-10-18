from torch._inductor.codegen.cpp import CppScheduling


class IpexCppScheduling(CppScheduling):
    def __init__(self, scheduler):
        super().__init__(scheduler)
