"""
Agent class for Newcastle traffic simulation.
Each agent represents a person with a daily schedule and demographic information.
Extends the base agent class with traffic-specific behavior.
"""

import os
import sys
from enum import Enum
from typing import Optional

import traci

# Add parent directory to path for base imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base.agent_base import AgentBase


class TransportMode(Enum):
    """Available transport modes"""

    CAR = "car"
    BUS = "bus"
    BICYCLE = "bicycle"
    WALK = "walk"
    METRO = "metro"  # Light rail


class ActivityType(Enum):
    """Types of activities agents can perform"""

    HOME = "home"
    WORK = "work"
    SCHOOL = "school"
    SHOPPING = "shopping"
    LEISURE = "leisure"


class Activity:
    """Represents a single activity in an agent's daily schedule"""

    def __init__(
        self,
        activity_type: ActivityType,
        location_edge: str,
        start_time: int,
        duration: int,
        transport_mode: TransportMode = TransportMode.CAR,
    ):
        self.type = activity_type
        self.location = location_edge
        self.start_time = start_time
        self.duration = duration
        self.end_time = start_time + duration
        self.completed = False
        self.transport_mode = transport_mode

    def __repr__(self):
        return f"Activity({self.type.value}, {self.location}, {self.start_time}-{self.end_time})"


class Agent(AgentBase):
    """
    Represents a single person in the traffic simulation.
    Manages daily schedule and interfaces with SUMO for movement.
    """

    def __init__(self, agent_id: str, demographics: dict, home_location: str):
        super().__init__(agent_id, demographics)
        self.home = home_location
        self.current_location = home_location

        # Schedule management (traffic-specific)
        self.schedule: list[Activity] = []
        self.current_activity_idx = 0

        # Trip state
        self.vehicle_id: Optional[str] = None
        self.in_transit = False

    def get_current_location(self) -> str:
        """Get the agent's current edge location"""
        return self.current_location

    def add_activity(self, activity: Activity):
        """Add a traffic-specific activity to the schedule"""
        self.schedule.append(activity)

    def get_current_activity(self) -> Optional[Activity]:
        """Get the current activity from the schedule"""
        if self.current_activity_idx < len(self.schedule):
            return self.schedule[self.current_activity_idx]
        return None

    def update(self, sim_time: int):
        """Main update logic called each simulation step"""
        current_activity = self.get_current_activity()

        if current_activity is None:
            return

        if sim_time >= current_activity.start_time and not self.in_transit:
            if not current_activity.completed:
                self._start_activity(current_activity, sim_time)

        if self.in_transit and self.vehicle_id:
            if self._has_arrived():
                self._complete_trip(sim_time)

        if (
            current_activity.completed
            and sim_time >= current_activity.end_time
            and not self.in_transit
        ):
            self._advance_to_next_activity()

    def _start_activity(self, activity: Activity, sim_time: int):
        """Begin an activity"""
        if self.current_location != activity.location:
            self._initiate_trip(
                self.current_location, activity.location, sim_time, activity.transport_mode
            )
        else:
            activity.completed = True

    def _initiate_trip(
        self,
        from_edge: str,
        to_edge: str,
        depart_time: int,
        mode: TransportMode = TransportMode.CAR,
    ):
        """Create a vehicle/person in SUMO and start the trip"""
        self.vehicle_id = f"{self.id}_t{depart_time}"
        self.in_transit = True

        if from_edge == to_edge:
            self.in_transit = False
            self.vehicle_id = None
            current_activity = self.get_current_activity()
            if current_activity:
                current_activity.completed = True
            return

        # Map transport mode to SUMO vehicle type ID
        # Use SUMO's built-in types where possible
        vtype_map = {
            TransportMode.CAR: "DEFAULT_VEHTYPE",  # SUMO built-in
            TransportMode.BUS: "pt_bus",  # From osm_pt.rou.xml
            TransportMode.BICYCLE: "DEFAULT_BIKETYPE",  # SUMO built-in
            TransportMode.WALK: "DEFAULT_PEDTYPE",  # SUMO built-in
            TransportMode.METRO: "pt_light_rail",  # From osm_pt.rou.xml
        }
        vtype = vtype_map.get(mode, "passenger_car")

        # Try to find route based on mode
        route_id = f"{self.vehicle_id}_route"
        try:
            if mode in [TransportMode.BUS, TransportMode.METRO]:
                # For public transport, use intermodal routing (includes walking)
                stage = traci.simulation.findIntermodalRoute(
                    from_edge,
                    to_edge,
                    modes="public",  # Enables walking + public transport
                    depart=depart_time,
                    pType="DEFAULT_PEDTYPE",
                )
            elif mode == TransportMode.WALK:
                # For walking, use intermodal with just walking
                stage = traci.simulation.findIntermodalRoute(
                    from_edge,
                    to_edge,
                    modes="",  # Empty = walking only
                    depart=depart_time,
                    pType="DEFAULT_PEDTYPE",
                )
            else:
                # For car/bicycle, use vehicle routing with specific type
                stage = traci.simulation.findRoute(from_edge, to_edge, vType=vtype)

            if not stage or (hasattr(stage, "edges") and not stage.edges):
                # No route found
                raise traci.exceptions.TraCIException("No route found")

        except traci.exceptions.TraCIException:
            # Route computation failed - edges not connected for this mode
            if self.diagnostics:
                self.diagnostics.teleports += 1
            self._teleport_to_destination(to_edge, depart_time)
            return

        # Now add vehicle/person with the computed route
        try:
            if mode == TransportMode.WALK:
                # For pedestrians, use person API with walking stage
                # findIntermodalRoute returns tuple of stages
                traci.person.add(self.vehicle_id, from_edge, 0.1)  # Start near edge start

                if isinstance(stage, tuple) and len(stage) > 0:
                    # Use first stage's edges (should be walking)
                    # Use negative value to mean "from end of edge" - SUMO accepts this
                    traci.person.appendWalkingStage(self.vehicle_id, stage[0].edges, -0.1)
                else:
                    traci.person.appendWalkingStage(self.vehicle_id, stage.edges, -0.1)

            elif mode in [TransportMode.BUS, TransportMode.METRO]:
                # For public transport users, create person with intermodal plan
                traci.person.add(self.vehicle_id, from_edge, 0.1)

                # findIntermodalRoute returns tuple of stages
                if isinstance(stage, tuple) and len(stage) > 0:
                    # Multi-stage intermodal trip (walk + PT + walk)
                    for substage in stage:
                        if substage.type == 2:  # Walking
                            # Use negative to count from end
                            traci.person.appendWalkingStage(self.vehicle_id, substage.edges, -0.1)
                        elif substage.type == 1:  # Driving (riding PT)
                            traci.person.appendDrivingStage(self.vehicle_id, to_edge, substage.line)
                else:
                    # Fallback to walking only
                    traci.person.appendWalkingStage(self.vehicle_id, [from_edge, to_edge], -0.1)
            else:
                # For vehicles (car, bicycle)
                traci.route.add(route_id, stage.edges)
                traci.vehicle.add(
                    vehID=self.vehicle_id,
                    routeID=route_id,
                    typeID=vtype,
                    depart="now",
                    departLane="free",
                    departPos="random",
                    departSpeed="max",
                )

            if self.diagnostics:
                self.diagnostics.trip_starts += 1

        except traci.exceptions.TraCIException as e:
            # Failed to insert - retry next step
            if self.diagnostics:
                self.diagnostics.failed_insertions += 1
                if self.diagnostics.failed_insertions <= 10:
                    print(
                        f"Failed to insert {self.id} on edge {from_edge} with mode {mode.value}: {e}"
                    )
            self.in_transit = False
            self.vehicle_id = None
            current_activity = self.get_current_activity()
            if current_activity:
                current_activity.completed = False

    def _teleport_to_destination(self, dest_edge: str, sim_time: int):
        """Teleport agent directly to destination when no route exists"""
        current_activity = self.get_current_activity()

        self.current_location = dest_edge
        self.in_transit = False
        self.vehicle_id = None

        if current_activity:
            current_activity.completed = True

    def _has_arrived(self) -> bool:
        """Check if the vehicle/person has reached its destination"""
        if not self.vehicle_id:
            return False

        try:
            # Check if it's a person (pedestrian)
            if self.vehicle_id in traci.person.getIDList():
                # For persons, check if they've completed their journey
                # getRemainingStages returns an integer count, not a list
                return traci.person.getRemainingStages(self.vehicle_id) == 0

            # Check if it's a vehicle
            if self.vehicle_id not in traci.vehicle.getIDList():
                return False

            road_id = traci.vehicle.getRoadID(self.vehicle_id)
            route = traci.vehicle.getRoute(self.vehicle_id)
            return road_id == route[-1] if route else False
        except traci.exceptions.TraCIException:
            return False

    def _complete_trip(self, sim_time: int):
        """Handle arrival at destination"""
        current_activity = self.get_current_activity()

        if current_activity:
            self.current_location = current_activity.location
        self.in_transit = False

        # Track completion
        if self.diagnostics:
            self.diagnostics.trip_completions += 1

        # CRITICAL: Remove vehicle/person from network immediately to free up road space
        try:
            if self.vehicle_id:
                if self.vehicle_id in traci.vehicle.getIDList():
                    traci.vehicle.remove(self.vehicle_id)
                elif self.vehicle_id in traci.person.getIDList():
                    traci.person.remove(self.vehicle_id)
        except traci.exceptions.TraCIException:
            pass

        self.vehicle_id = None

        if current_activity:
            current_activity.completed = True

    def _advance_to_next_activity(self):
        """Move to the next activity in the schedule"""
        self.current_activity_idx += 1

    def replan_schedule(self, event: dict):
        """Generate new schedule in response to an event"""
        pass
