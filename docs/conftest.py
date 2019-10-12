# PyTrain — Copyright (c) 2019, Alex J. Champandard.

from doctest import ELLIPSIS
from sybil import Sybil
from sybil.parsers.codeblock import CodeBlockParser
from sybil.parsers.doctest import DocTestParser
from sybil.parsers.skip import skip
from sybil.parsers.capture import parse_captures

pytest_collect_file = Sybil(
    parsers=[
        CodeBlockParser(),
        DocTestParser(optionflags=ELLIPSIS),
        parse_captures,
        skip,
    ],
    pattern="*.rst",
).pytest()
