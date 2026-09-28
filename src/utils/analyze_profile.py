import pstats

stats = pstats.Stats("profile_v2.prof")

stats.strip_dirs()
stats.sort_stats("cumulative")

stats.print_stats(30)