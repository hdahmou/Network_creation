"""Canonical observer name-to-class mapping."""

from __future__ import annotations

from .action_count_observer import ActionCountObserver
from .action_print_observer import ActionPrintObserver
from .central_point_of_dominance_observer import CentralPointOfDominanceObserver
from .density_observer import DensityObserver
from .distance_observer import DistanceObserver
from .global_closeness_centrality_observer import (
    GlobalClosenessCentralityObserver,
)
from .hub_persistence_observer import HubPersistenceObserver
from .indegree_observer import IndegreeObserver
from .social_welfare_observer import SocialWelfareObserver
from .state_history_observer import StateHistoryObserver
from .triad_census_observer import TriadCensusObserver


OBSERVER_GRIMOIRE = {
    "action_count": ActionCountObserver,
    "action_print": ActionPrintObserver,
    "central_point_of_dominance": CentralPointOfDominanceObserver,
    "density": DensityObserver,
    "distance": DistanceObserver,
    "global_closeness_centrality": GlobalClosenessCentralityObserver,
    "hub_persistence": HubPersistenceObserver,
    "indegree": IndegreeObserver,
    "social_welfare": SocialWelfareObserver,
    "state_history": StateHistoryObserver,
    "triad_census": TriadCensusObserver,
}


__all__ = ["OBSERVER_GRIMOIRE"]
