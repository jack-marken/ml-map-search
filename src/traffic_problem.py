from src.data_structures import Site, Link
from src.traffic_utils import flow_to_speed, haversine_distance
import numpy as np
import datetime

class TrafficProblem():
    """
    The Traffic-Based Route Guidance Problem.

    Contains all sites, intersections, and connecting links.
    An origin and destination site are set upon initialisation.
    """
    def __init__(self, sites, intersections, origin, destination, links, date_time=datetime.datetime(2006, 10, 1, 0, 0), estimator=None):
        self.sites = sites # [<Site>, <Site>, ...] - All sites in the problem
        self.intersections = intersections # [<Intersection>, <Intersection>, ...] - All intersections in the problem
        self.origin = next(s for s in self.sites if s.scats_num == origin)    # <Site> - first site of the search
        self.destination = next(s for s in self.sites if s.scats_num == destination)        # <Site> - the final site of the search
        self.links = links # [<Link>, <Link>, ...]
        self.date_time = date_time # the current time
        self.estimator = estimator  # TravelTimeEstimator instance
  
    def get_site_by_scats(self, scats_num):
        for site in self.sites:
            if site.scats_num == scats_num:
                return site
        return None

    def get_site_by_intersection(self, intersection):
        for site in self.sites:
            if intersection in site.intersections:
                return site
        return None

    def get_actions(self, s):
        actions = []
        for l in self.links:
            if l.origin.scats_num == s.scats_num:
                actions.append(self.get_site_by_intersection(l.destination))
        return actions

    # Returns a bool: is site 's' the destination?
    def goal_test(self, s):
        return s == self.destination

    def distance_heuristic(self, s):
        """
        Computes the Haversine (great-circle) distance, in km, of the closest intersections between site 's' and the destination
        """
        if self.goal_test(s):
            return 0

        min_dist = float('inf')
        for site_i in s.intersections: # Intersections of site 's'
            for dest_i in self.destination.intersections: # Intersections of the destination site
                dist = haversine_distance(site_i.coordinates[0], site_i.coordinates[1], dest_i.coordinates[0], dest_i.coordinates[1])
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def distance_between_sites(self, a, b):
        """
        Computes the Haversine (great-circle) distance, in km, of the closest intersections between site 'a' and site 'b'
        """
        min_dist = float('inf')
        for site_i in a.intersections: # Intersections of site 'a'
            for dest_i in b.intersections: # Intersections of the destination site
                dist = haversine_distance(site_i.coordinates[0], site_i.coordinates[1], dest_i.coordinates[0], dest_i.coordinates[1])
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def get_flow_at_time(self, s, date_hour, ml_model):
        """
        Given site 's' and a datetime.time object,
        return the flow closest to the provided time.
        """
        # Return the average of all flow records of site 's' at the date/hour 'date_hour'
        summed_flow = 0.0
        records_counted = 0
        get_keys_within_the_same_hour = lambda dataset: [record for record in dataset.keys() if record.date() == date_hour.date() and record.hour == date_hour.hour]

        for intersection in s.intersections:
            flow = 0
            match ml_model:
                case None:
                    for key in get_keys_within_the_same_hour(intersection.flow_records):
                        flow += intersection.flow_records[key]
                case 'LSTM':
                    for key in get_keys_within_the_same_hour(intersection.lstm_data):
                        flow += intersection.lstm_data[key]
                case 'GRU':
                    for key in get_keys_within_the_same_hour(intersection.gru_data):
                        flow += intersection.gru_data[key]
                case 'RNN':
                    for key in get_keys_within_the_same_hour(intersection.rnn_data):
                        flow += intersection.rnn_data[key]
                        
            summed_flow += float(flow)
            records_counted += 1
        if records_counted > 0:
            avg_flow = summed_flow / records_counted
            return avg_flow
        return 0

    def travel_time(self, a, b, ml_model):
        """
        Calculate travel time from Site a to Site b using flow data and haversine distance
        """
        dist = self.distance_between_sites(a,b) # Distance (km) between site 'a' and site 'b'
        
        flow = self.get_flow_at_time(b, self.date_time, ml_model)
        speed = flow_to_speed(flow) # in km/h

        travel_seconds = (dist / speed) * 3600 + 30 # add delay
        return travel_seconds
