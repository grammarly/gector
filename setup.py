from setuptools import find_packages, setup


def requirements():
    req_path = 'requirements.txt'
    with open(req_path) as f:
        reqs = f.read().splitlines()
    return reqs


setup(
    name='gector',
    url='https://github.com/grammarly/gector',
    author='grammarly',
    packages=find_packages(),
    version='1.0',
    license='Apache-2.0 License',
    description='GECToR – Grammatical Error Correction: Tag, Not Rewrite',
    long_description=open('README.md').read(),

    install_requires=requirements(),
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved :: Apache-2.0 License',
                 'Programming Language :: Python',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: Unix',
                 'Programming Language :: Python :: 3.8',
                 ]
)
