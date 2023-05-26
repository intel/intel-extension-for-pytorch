# examples
# mode 0
# 0 p0 0 | 4 p4 1
# 1 p1 0 | 5 p5 1
# 2 p2 0 | 6 p6 1
# 3 p3 0 | 7 p7 1

# 0 p0 0 | 4 l0 0 |  8 p4 1 | 12 l4 1
# 1 p1 0 | 5 l1 0 |  9 p5 1 | 13 l5 1
# 2 p2 0 | 6 l2 0 | 10 p6 1 | 14 l6 1
# 3 p3 0 | 7 l3 0 | 11 p7 1 | 15 l7 1

# mode 1
# 0 p0 0 | 4 p4 1
# 1 p1 0 | 5 p5 1
# 2 p2 0 | 6 p6 1
# 3 p3 0 | 7 p7 1

# 0 p0 0 | 4 p4 1 |  8 l0 0 | 12 l4 1
# 1 p1 0 | 5 p5 1 |  9 l1 0 | 13 l5 1
# 2 p2 0 | 6 p6 1 | 10 l2 0 | 14 l6 1
# 3 p3 0 | 7 p7 1 | 11 l3 0 | 15 l7 1

# mode 2
# 0 p0 0 | 4 p4 1
# 1 p1 0 | 5 p5 1
# 2 p2 0 | 6 p6 1
# 3 p3 0 | 7 p7 1

# 0 p0 0 | 4 p2 0 |  8 p4 1 | 12 p6 1
# 1 l0 0 | 5 l2 0 |  9 l4 1 | 13 l6 1
# 2 p1 0 | 6 p3 0 | 10 p5 1 | 14 p7 1
# 3 l1 0 | 7 l3 0 | 11 l5 1 | 15 l7 1


class CoreInfo:
    def __init__(
        self,
        cpu=-1,
        core=-1,
        socket=-1,
        node=-1,
        is_physical_core=True,
        maxmhz=0.0,
        is_p_core=True,
    ):
        self.cpu = cpu
        self.core = core
        self.socket = socket
        self.node = node
        self.is_physical_core = is_physical_core
        self.maxmhz = maxmhz
        self.is_p_core = is_p_core

    def __str__(self):
        return f"{self.cpu}\t{self.core}\t{self.socket}\t{self.node}\t{self.is_physical_core}\t{self.maxmhz}\t{self.is_p_core}"


def construct_numa_config(
    n_nodes,
    n_phycores_per_node,
    enable_ht=True,
    n_e_cores=0,
    numa_mode=0,
    show_node=True,
):
    cores = []
    for i in range(n_nodes):
        for j in range(n_phycores_per_node):
            core_id = i * (n_phycores_per_node + n_e_cores) + j
            cores.append(CoreInfo(core=core_id, socket=i, node=i, maxmhz=5000.0))
            if enable_ht:
                cores.append(
                    CoreInfo(
                        core=core_id,
                        socket=i,
                        node=i,
                        is_physical_core=False,
                        maxmhz=5000.0,
                    )
                )
        for j in range(n_e_cores):
            core_id = i * (n_phycores_per_node + n_e_cores) + n_phycores_per_node + j
            cores.append(
                CoreInfo(core=core_id, socket=i, node=i, maxmhz=3800.0, is_p_core=False)
            )
    if numa_mode == 0:
        cores.sort(
            key=lambda x: (
                x.node,
                1 - int(x.is_p_core),
                1 - int(x.is_physical_core),
                x.core,
            )
        )
    if numa_mode == 1:
        cores.sort(
            key=lambda x: (
                1 - int(x.is_p_core),
                1 - int(x.is_physical_core),
                x.node,
                x.core,
            )
        )
    if numa_mode == 2:
        cores.sort(
            key=lambda x: (
                x.node,
                1 - int(x.is_p_core),
                x.core,
                1 - int(x.is_physical_core),
            )
        )
    for i in range(len(cores)):
        cores[i].cpu = i
    cores.sort(key=lambda x: x.cpu)
    ret = []
    if show_node:
        ret.append("CPU CORE SOCKET NODE MAXMHZ")
    else:
        ret.append("CPU CORE SOCKET MAXMHZ")
    for c in cores:
        if show_node:
            ret.append(f"{c.cpu} {c.core} {c.socket} {c.node} {c.maxmhz}")
        else:
            ret.append(f"{c.cpu} {c.core} {c.socket} {c.maxmhz}")
    return "\n".join(ret)


if __name__ == "__main__":
    lscpu_txt = construct_numa_config(
        n_nodes=1,
        n_phycores_per_node=8,
        enable_ht=True,
        n_e_cores=8,
        numa_mode=2,
        show_node=True,
    )
    lscpu_txt = lscpu_txt.replace(" ", "\t")
    print(lscpu_txt)
