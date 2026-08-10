def manhattan_distance(pos_a: tuple[int, int], pos_b: tuple[int, int]) -> int:
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])