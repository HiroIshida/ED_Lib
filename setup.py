try:
    from skbuild import setup
except ImportError:
    raise Exception

setup(
    name="edlib",
    version="0.0.0",
    description="edlib python binding",
    install_requires=["numpy"],
    packages=["edlib"],
    package_dir={"": "python"},
    package_data={"edlib": ["__init__.pyi"]},
    cmake_install_dir="python/edlib/",
)
