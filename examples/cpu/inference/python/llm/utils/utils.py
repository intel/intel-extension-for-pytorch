import re


def _get_relative_imports(module_file):
    with open(module_file, "r", encoding="utf-8") as f:
        content = f.read()
    relative_imports = re.findall(
        r"^\s*import\s+\.(\S+)\s*$", content, flags=re.MULTILINE
    )
    relative_imports += re.findall(
        r"^\s*from\s+\.(\S+)\s+import", content, flags=re.MULTILINE
    )
    relative_imports = set(relative_imports)
    # For Baichuan2
    if "quantizer" in relative_imports:
        relative_imports.remove("quantizer")

    return list(relative_imports)