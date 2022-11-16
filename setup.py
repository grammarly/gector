from setuptools import find_packages, setup

# Get version without importing, which avoids dependency issues
exec(compile(open("gector/version.py").read(), "gector/version.py", "exec"))
# (we use the above instead of execfile for Python 3.x compatibility)


def requirements():
    req_path = "requirements.txt"
    with open(req_path) as f:
        reqs = f.read().splitlines()
    return reqs


setup(
    name="gector",
    url="https://github.com/grammarly/gector",
    author="grammarly",
    packages=find_packages(),
    version=__version__,
    license="Apache-2.0 License",
    description="GECToR – Grammatical Error Correction: Tag, Not Rewrite",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    install_requires=requirements(),
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
