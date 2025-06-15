from .search_method import SearchMethod

class BS(SearchMethod):
    name = "BS"

    def search(self, beam_width=2):
        h = self.problem.distance_heuristic

        while self.frontier:
            current_site, path = self.frontier.pop()
            path = path + [current_site]
            self.explored.append(current_site)

            if self.problem.goal_test(current_site):
                self.result = current_site
                self.final_path = path
                for i in range(len(self.final_path) - 1):
                    self.travel_time += self.problem.travel_time(self.final_path[i], self.final_path[i+1], self.ml_model)
                return

            ## A list of connected sites (actions) sorted by the shortest distance to the nearest destination
            actions_sorted_by_id = [site for site in sorted(self.problem.get_actions(current_site), key=lambda x: x.scats_num, reverse=True)]
            actions = [site for site in sorted(actions_sorted_by_id, key=lambda x: h(x), reverse=True)]
            for site in actions[-beam_width:]:
                if not site in self.explored:
                    self.frontier.append((site, path))

            ################
            # self.print_state(current_site, actions) # <-- For debugging only
            ################
