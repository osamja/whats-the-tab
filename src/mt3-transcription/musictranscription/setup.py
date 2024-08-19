from setuptools import setup

setup(
    name='your_project_name',
    version='0.1',
    install_requires=[
        'jax==0.4.28',
        'nest-asyncio',
        'pyfluidsynth==1.3.0',
        '-e .'
    ],
    dependency_links=[
        'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
    ],
    extras_require={
        'cuda': [
            'jaxlib==0.4.28+cuda12.cudnn89'
        ]
    }
)
