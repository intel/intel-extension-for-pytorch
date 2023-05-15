import subprocess


def get_num_nodes():
    return int(
        subprocess.check_output(
            "LANG=C; lscpu | grep Socket | awk '{print $2}'", shell=True
        )
    )


def get_num_cores_per_node():
    return int(
        subprocess.check_output(
            "LANG=C; lscpu | grep Core'.* per socket' | awk '{print $4}'", shell=True
        )
    )


def get_core_list_of_node_id(node_id):
    r"""
    Helper function to get the CPU cores' ids of the input numa node.

    Args:
        node_id (int): Input numa node id.

    Returns:
        list: List of CPU cores' ids on this numa node.
    """

    num_of_nodes = get_num_nodes()
    assert (
        node_id < num_of_nodes
    ), "input node_id:{0} must less than system number of nodes:{1}".format(
        node_id, num_of_nodes
    )
    num_cores_per_node = get_num_cores_per_node()
    return list(range(num_cores_per_node * node_id, num_cores_per_node * (node_id + 1)))
