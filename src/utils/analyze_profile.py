import pstats

stats = pstats.Stats("profiling.prof")

stats.strip_dirs()
stats.sort_stats("cumulative")

stats.print_stats(30)