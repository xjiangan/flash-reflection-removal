import pstats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file")
parser.add_argument("--sort",default="time")
parser.add_argument("--num",type=int,default=10)


ARGS = parser.parse_args()
p=pstats.Stats(ARGS.file)
print(p.sort_stats(ARGS.sort).print_stats(ARGS.num))