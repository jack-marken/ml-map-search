from .file_parser import FileParser
from .traffic_problem import TrafficProblem
from .search_methods import SearchMethod, DFS, BFS, GBFS, AS, IDDFS, BS

ALGORITHMS = {
    "DFS": DFS,
    "BFS": BFS,
    "GBFS": GBFS,
    "AS": AS,
    "IDDFS": IDDFS,
    "BS": BS
}
