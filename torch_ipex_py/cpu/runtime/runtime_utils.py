import subprocess

def get_sockets():
    return int(subprocess.check_output('lscpu | grep Socket | awk \'{print $2}\'', shell=True))

def get_cores_per_socket():
    return int(subprocess.check_output('lscpu | grep Core | awk \'{print $4}\'', shell=True))

def get_core_list_of_socket_id(socket_id):
    sockets = get_sockets()
    assert socket_id < sockets, "input socket_id:{0} must less than system sockets:{1}".format(socket_id, sockets)
    cores_per_socket = get_cores_per_socket()
    return list(range(cores_per_socket * socket_id, cores_per_socket * (socket_id + 1)))
