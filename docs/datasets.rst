Datasets
========

Tasks often require data that's provided by external datasets, and they are specified to ``pytrain`` using **type annotations**.  In this case, ``MyData`` is a constructor that's used to create the dataset, then automatically populates the ``batch``:

.. These constructors are defined later in one of many different ways.
.. invisible-code-block: python
    MyData = None
    MyModel = None

.. code-block:: python

    def task_example(batch: MyData, model: MyModel):
        output = model(batch.data)
        return torch.nn.functional.mse_error(output, batch.target)

There are multiple ways that ``MyData`` can be defined and understood by ``pytrain``.


Array Constructor
-----------------

One of the simplest ways to define a dataset is simply to create a function that returns an array in the format. In this case, there are 192 items of 2D data, specifically 3 channels and 16 samples:

.. code-block:: python

    def MyData():
        return torch.zeros(size=(192,3,16), dtype=torch.float)

This function acts as a constructor to create data that ``pytrain`` can convert into a ``Dataset`` internally.

>>> MyData
<function MyData at ...>


Array Object
------------

If you need more control over the way items are fetched from the dataset, for example to load files from disk on-the-fly, you can define an array-like object as a Python class.

.. code-block:: python

    class MyData:

        def __init__(self):
            self._data = torch.zeros(size=(192,3,16), dtype=torch.float)

        def __len__(self):
            return self._data.shape[0]

        def __getitem__(self, idx):
            return self._data[idx]

In this case, the class is instanciated by ``pytrain`` to create an object that will be accessed as a ``Dataset`` internally.

>>> MyData
<class 'MyData'>


Iterator Function
-----------------

Another alternative to provide data is to simply create an iterator that infinitely returns new ``Batch`` objects.  Note that ``pytrain`` is not able to customize the iteration order (e.g. to prioritize the training) this way.

.. code-block:: python

    def MyData():
        _data = torch.empty((192, 3, 16), dtype=torch.float).uniform_()
 
        while True:
            yield pytrain.Batch(data=_data)

>>> MyData
<function MyData at ...>


Iterator Object
---------------

Combining the options above, you can also create an object that acts like an iterator.  It looks like this:

.. code-block:: python

    class MyData:
        def __init__(self):
            self._data = torch.empty((192, 3, 16), dtype=torch.float).uniform_()

        def __next__(self):
            idx = torch.rand.randint(0, 192)
            yield pytrain.Batch(data=self._data[idx])

>>> MyData
<class 'MyData'>


Dataset Constructor
-------------------

Finally, for the most flexibility you can create a ``Dataset`` object exactly as is used by ``pytrain`` internally.  Each of the ``training``, ``validation`` and ``testing`` values can be iterators, constructors, or objects as explained in the previous sections.

.. code-block:: python

    def MyData():
        t = torch.zeros(size(128,3,16), dtype=torch.float)
        v = torch.zeros(size(64,3,16), dtype=torch.float)
        return pytrain.Dataset(training=t, validation=v)

>>> MyData
<function MyData at ...>
