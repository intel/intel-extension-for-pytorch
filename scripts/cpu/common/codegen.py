import os

def write_or_skip(filepath, content):
    try:
        with open(filepath, 'r') as f:
            old_content = f.read()
    except IOError:
        old_content = None

    if old_content != content:
        with open(filepath, 'w') as f:
            print('writing', filepath)
            f.write(content)
    else:
        print('skipped writing', filepath)