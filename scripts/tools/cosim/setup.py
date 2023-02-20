from setuptools import setup
import distutils.command.clean
import os
import shutil
import glob

class CosimClean(distutils.command.clean.clean, object):
    def clean_dir(self, d):
        if not os.path.exists(d):
            print(f"removing '{d}' (skipped as it does not exist)")
        else:
            for filename in glob.glob(d):
                try:
                    os.remove(filename)
                    print(f"removing '{filename}'")
                except OSError:
                    shutil.rmtree(filename, ignore_errors=True)
                    print(f"removing '{filename}' (and everything under it)")

    def run(self):
        self.clean_dir('build')
        self.clean_dir('cosim.egg-info')
        self.clean_dir('dist')
        distutils.command.clean.clean.run(self)


cmdclass = {
    'clean': CosimClean,
}

setup(
    name='cosim',
    version='1.0',
    description='cosim plugin',
    packages=['cosim'],
    install_requires=['torch', 'intel_extension_for_pytorch'],
    package_data={'': ['*']},
    cmdclass=cmdclass,
    include_package_data=True,
)
