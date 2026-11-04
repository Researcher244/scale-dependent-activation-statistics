"""Print environment information without importing training code."""
import importlib.metadata
import platform
import sys

print('Python:',sys.version)
print('Platform:',platform.platform())
for name in ['numpy','pandas','scipy','matplotlib','torch','torchvision','scikit-learn','notebook','jupyterlab']:
    try: version=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:version='not installed'
    print(f'{name}: {version}')
