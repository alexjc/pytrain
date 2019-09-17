.. image:: docs/logo.png

Like ``pytest`` supports you in writing better Python code through automated testing, ``pytrain`` helps you build differentiable programs by making it easy to create tasks and use them optimize your programs.


Installation
============

.. code:: bash

    # Create a base environment
    conda create -n myenv python=3.6
    conda install pytorch -c pytorch

    # Either install latest package:
    pip install https://github.com/alexjc/pytrain/releases/download/v0.0.1/pytrain-0.0.1.tar.gz

    # Or clone the repository online:
    git clone https://github.com/alexjc/pytrain.git

Usage
=====

.. code:: bash

    # Either launch from installed script:
    pytrain -h
    pytrain -r examples/

    # Or run from current directory:
    python -m pytrain -h
    python -m pytrain -r examples/


Examples
========

See the ``#examples/`` folder to get up and running.

*NOTE: This version 0.0.1 is an early prototype for the PyTorch Hackathon 2019.  Feedback and suggestions are the most welcome at this stage!*

.. image:: docs/console.gif
