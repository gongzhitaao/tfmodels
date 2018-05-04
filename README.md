Tensorflow Models
=================

> What I cannot create, I do not understand.  -- Feynman

This repo contains TensorFlow re-implementation of interesting models from
random papers I read.  I use the low-level API as much as possible since
research models almost never fit into the standard API.

## Getting Started

Personally I use [*editable installs*](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs) to install the modules locally in a
virtual environment created with [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) since the changes made
locally will be readily available in local environment without re-installing the
package.

```bash
python install -e path/to/tfmodels
```

It should be also be easy to integrate code from this repo to your project
simply by copy-and-paste.

## Model List

- Text classification.  Word-level CNN models, char-level model.

## Related

- [tensorflow/models](https://github.com/tensorflow/models)
