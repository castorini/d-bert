import sys


init_seed = int(sys.argv[2])
for x in range(int(sys.argv[1])):
    l = x % 2
    print(f'./{sys.argv[3]} {l} {init_seed} {x % 8} &')
    init_seed += 1

