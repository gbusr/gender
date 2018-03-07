import json
import tensorflow as tf


class ParseConfig(object):
    def __init__(self, json_file):
        self.file = json_file
        self.input = self._get_config_input()
        self.optimzer = self._get_config_optimzer()
        self.log = self._get_config_log()

    def _parse_json(self, root):
        with open(self.file, 'r') as load_f:
            load_dict = json.load(load_f)
            load_root = load_dict[root]
        return load_root[0]

    def _get_config_input(self):
        return self._parse_json("input")

    def _get_config_optimzer(self):
        return self._parse_json("optimzer")

    def _get_config_log(self):
        return self._parse_json("log")


def configure_session():
    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    return tf.ConfigProto(allow_soft_placement=True,
                          gpu_options=gpu_config)
