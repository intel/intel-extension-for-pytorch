import unittest
from common_utils import VerboseTestCase
import subprocess


class TestLinearReorder(VerboseTestCase):
    def test_linear_reorder(self):
        with subprocess.Popen(
            "DNNL_VERBOSE=1 python -u linear_reorder.py",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as p:
            segmentation = {
                "fp32": {
                    "reorder_for_pack": 2,
                    "reorder_for_dtype": 0,
                    "reorder_for_format": 0,
                    "redundent_reorder": 0,
                },
                "bf16": {
                    "reorder_for_pack": 3,
                    "reorder_for_dtype": 0,
                    "reorder_for_format": 0,
                    "redundent_reorder": 0,
                },
            }  # there should be only reorders on prepack, if any other reorder appears, will cause fail
            seg = None
            for line in p.stdout.readlines():
                line = str(line, "utf-8").strip()
                if line.endswith("***************"):
                    seg = line.strip().split(",")[0]
                    continue
                # Following is to check if there is the reorder number is as excepted
                if self.is_dnnl_verbose(line) and self.ReorderForPack(line):
                    segmentation[seg]["reorder_for_pack"] -= 1
                    self.assertTrue(
                        segmentation[seg]["reorder_for_pack"] >= 0,
                        "show unexpected reorder for pack",
                    )

                if self.is_dnnl_verbose(line) and self.OnlyReorderDtype(line):
                    segmentation[seg]["reorder_for_dtype"] -= 1
                    self.assertTrue(
                        segmentation[seg]["reorder_for_dtype"] >= 0,
                        "show unexpected reorder for dtype",
                    )

                if self.is_dnnl_verbose(line) and self.OnlyReorderFormat(line):
                    segmentation[seg]["reorder_for_format"] -= 1
                    self.assertTrue(
                        segmentation[seg]["reorder_for_format"] >= 0,
                        "show unexpected reorder for format",
                    )

                if self.is_dnnl_verbose(line) and self.RedundantReorder(line):
                    segmentation[seg]["redundent_reorder"] -= 1
                    self.assertTrue(
                        segmentation[seg]["redundent_reorder"] >= 0,
                        "show unexpected redundent reorder",
                    )


if __name__ == "__main__":
    test = unittest.main()
