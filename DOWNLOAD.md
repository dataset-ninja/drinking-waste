Dataset **Drinking Waste** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/V/U/8M/D5d4uB69s5AFEKnUXvzIqa9jCGW1J7oVILx6zx10ug91MV5dUsI5EIqVqnysfqenutJINOspMixSUsNDFJ71dtfF2bGm1TvAqF1JP2vtXsUuhTqcquAPCHq3RJmz.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='Drinking Waste', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be [downloaded here](https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification/download?datasetVersionNumber=2).