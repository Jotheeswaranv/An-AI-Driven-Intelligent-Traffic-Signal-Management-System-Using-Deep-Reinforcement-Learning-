"""
traffic_logic.py
Advanced signal decision engine.
Handles:
  - Single ambulance: immediate green to that lane
  - Multiple ambulances: green to lane with highest density
  - Normal: green to highest density lane
  - Green time proportional to density
"""
from app import logger

BASE_GREEN_TIME    = 10
MAX_GREEN_TIME     = 60
MIN_GREEN_TIME     = 5
DENSITY_TIME_FACTOR = 2.0


def compute_signal_plan(lane_results: dict) -> dict:
    plan = {}
    if not lane_results:
        return plan

    # Find all lanes with emergency
    emergency_lanes = {
        lane: data for lane, data in lane_results.items()
        if data.get('has_emergency', False)
    }

    if emergency_lanes:
        if len(emergency_lanes) == 1:
            # Single ambulance → immediate green to that lane
            priority_lane = list(emergency_lanes.keys())[0]
            logger.info(f'SINGLE EMERGENCY: {priority_lane} → GREEN')
            for lane in lane_results:
                if lane == priority_lane:
                    plan[lane] = {
                        'signal': 'GREEN',
                        'green_time': MAX_GREEN_TIME,
                        'reason': '🚨 Emergency Vehicle — Priority Override'
                    }
                else:
                    plan[lane] = {
                        'signal': 'RED',
                        'green_time': 0,
                        'reason': 'RED — Emergency in another lane'
                    }

        else:
            # Multiple ambulances → green to highest density emergency lane
            densities = {
                lane: data.get('density', 0)
                for lane, data in emergency_lanes.items()
            }
            priority_lane = max(densities, key=densities.get)
            logger.info(
                f'MULTIPLE EMERGENCIES: {list(emergency_lanes.keys())} '
                f'→ {priority_lane} wins (density={densities[priority_lane]})'
            )
            for lane in lane_results:
                if lane == priority_lane:
                    plan[lane] = {
                        'signal': 'GREEN',
                        'green_time': MAX_GREEN_TIME,
                        'reason': (
                            f'🚨 Multi-Emergency — Highest density '
                            f'({densities[priority_lane]})'
                        )
                    }
                elif lane in emergency_lanes:
                    plan[lane] = {
                        'signal': 'RED',
                        'green_time': 0,
                        'reason': '🚨 Emergency — waiting (lower density)'
                    }
                else:
                    plan[lane] = {
                        'signal': 'RED',
                        'green_time': 0,
                        'reason': 'RED — Emergency override active'
                    }
        return plan

    # Normal density-based logic
    densities = {
        lane: data.get('density', 0)
        for lane, data in lane_results.items()
    }
    max_lane    = max(densities, key=densities.get) if densities else None
    max_density = densities.get(max_lane, 0)

    for lane, density in densities.items():
        if lane == max_lane and max_density > 0:
            green_time = int(BASE_GREEN_TIME + density * DENSITY_TIME_FACTOR)
            green_time = max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, green_time))
            plan[lane] = {
                'signal': 'GREEN',
                'green_time': green_time,
                'reason': f'Highest density ({density}) — normal priority'
            }
        else:
            plan[lane] = {
                'signal': 'RED',
                'green_time': 0,
                'reason': f'Density {density} — waiting'
            }

    logger.info(f'Normal plan: green={max_lane}, densities={densities}')
    return plan
