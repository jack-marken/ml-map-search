from .search_method import SearchMethod

class IDDFS(SearchMethod):
    name = "IDDFS"

    def __init__(self, problem, ml_algorithm=None):
        super().__init__(problem, ml_algorithm)
        self.frontier = [(self.problem.origin, [], 0)] # Same as parent class but containing a third 'depth' value

    def search(self):
        depth = 0
        while True:
            if self.depth_limited_dfs(depth):
                return
            depth += 1
            if depth > 50:
                return

    def depth_limited_dfs(self, depth_limit): # also known as Depth Limited Search (DLS)
        depth = 0
        local_explored = []
        
        while self.frontier:
            current_site, path, depth = self.frontier.pop()
            path = path + [current_site]
            local_explored.append(current_site)

            if self.problem.goal_test(current_site):
                self.result = current_site
                self.explored += [current_site]
                self.final_path = path
                for i in range(len(self.final_path) - 1):
                    self.travel_time += self.problem.travel_time(self.final_path[i], self.final_path[i+1], self.ml_model)
                return True

            if depth > depth_limit:     
                self.frontier = [(self.problem.origin, [], 0)] # Reset frontier to be ready for the next depth
                self.explored += local_explored                # self.explored will hold the paths of all DLS iterations
                return False

            actions = [site for site in reversed(sorted(self.problem.get_actions(current_site), key=lambda x: x.scats_num))]
            depth += 1
            for site in actions:
                if not site in local_explored:
                    self.frontier.append((site, path, depth))

            ################
            # self.print_state(node, actions) # <-- For debugging only
            ################
        return False
