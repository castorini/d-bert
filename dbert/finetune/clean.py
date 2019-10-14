import re
import sys


patt = re.compile(r'^\w[^\t]*\s*?\t\s*?\w[^\t]*$')


for x in sys.stdin:
    if re.match(patt, x):
        print(x, end='')
