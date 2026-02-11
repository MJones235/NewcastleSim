#!/usr/bin/env python3
"""
Generate MP4 video from Station Concordia simulation output.

This script creates a video from saved simulation data, showing
agent positions and decisions at regular time intervals without
delays for LLM responses.

Usage:
    python tools/generate_video.py --output-file PATH --video-file PATH
    python tools/generate_video.py --output-file scenarios/station_concordia/output/run_20240210_120000/agent_decisions.json

Requirements:
    - matplotlib
    - ffmpeg (must be installed on system)
"""

import argparse
import json
import sys
from pathlib import Path

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Module imports after path setup
from scenarios.common.logger import get_logger  # noqa: E402
from scenarios.station_concordia.visualization.video_generator import (  # noqa: E402
    generate_video_from_output,
)

logger = get_logger(__name__)


def load_geometry_from_network(network_path: Path) -> dict | None:
    """Load station geometry from SUMO network."""
    try:
        from scenarios.station_jupedsim.geometry import (
            load_entrance_areas,
            load_obstacles,
            load_platform_areas,
            load_walkable_areas,
        )

        walking_areas_file = network_path / "walking_areas.add.xml"
        if not walking_areas_file.exists():
            logger.warning(f"Geometry file not found: {walking_areas_file}")
            return None

        walkable_areas = load_walkable_areas(str(walking_areas_file))
        entrance_areas = load_entrance_areas(str(walking_areas_file))
        platform_areas = load_platform_areas(str(walking_areas_file))
        obstacles = load_obstacles(str(walking_areas_file))

        def poly_to_coords(poly):
            return list(poly.exterior.coords)

        geometry = {
            "walkable_areas": {name: poly_to_coords(poly) for name, poly in walkable_areas.items()},
            "entrance_areas": {name: poly_to_coords(poly) for name, poly in entrance_areas.items()},
            "platform_areas": {name: poly_to_coords(poly) for name, poly in platform_areas.items()},
            "obstacles": [poly_to_coords(poly) for poly in obstacles],
        }

        logger.info(
            f"Loaded geometry: {len(walkable_areas)} walkable areas, "
            f"{len(entrance_areas)} entrances, {len(platform_areas)} platforms, "
            f"{len(obstacles)} obstacles"
        )

        return geometry
    except Exception as e:
        logger.error(f"Failed to load geometry: {e}")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate video from Station Concordia simulation output"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to agent decisions JSON file",
    )
    parser.add_argument(
        "--video-file",
        type=str,
        default=None,
        help="Output video path (default: same directory as output file)",
    )
    parser.add_argument(
        "--network-path",
        type=str,
        default="scenarios/station_sim/network",
        help="Path to station network directory (for geometry)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video frames per second (default: 20)",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Video speed multiplier (default: 1.0 = real-time)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Video resolution in DPI (default: 100, higher = better quality)",
    )

    args = parser.parse_args()

    output_file = Path(args.output_file)
    if not output_file.exists():
        logger.error(f"Output file not found: {output_file}")
        sys.exit(1)

    # Load geometry
    network_path = Path(args.network_path)
    geometry = load_geometry_from_network(network_path)

    # Check for position history
    history_file = output_file.parent / f"{output_file.stem}_history.json"
    data_file = output_file

    if history_file.exists():
        logger.info(f"Found position history: {history_file}")

        # Merge position history with decisions data
        with open(history_file) as f:
            history_data = json.load(f)
        with open(output_file) as f:
            decisions_data = json.load(f)

        decisions_data["position_history"] = history_data.get("position_history", [])

        # Save merged data temporarily
        data_file = output_file.parent / f"{output_file.stem}_merged_temp.json"
        with open(data_file, "w") as f:
            json.dump(decisions_data, f)

        logger.info(
            f"Merged position history ({len(history_data.get('position_history', []))} frames)"
        )
    else:
        logger.warning(
            "No position history found - video will show final state only. "
            "Enable 'video.enabled: true' in config.yaml during simulation "
            "to track positions for full animation."
        )

    # Determine video output path
    if args.video_file:
        video_path = Path(args.video_file)
    else:
        video_path = output_file.parent / f"{output_file.stem}_video.mp4"

    # Generate video
    logger.info("=" * 60)
    logger.info(f"Generating video: {video_path}")
    logger.info(f"Settings: {args.fps} fps, {args.speedup}x speed, {args.dpi} dpi")
    logger.info("=" * 60)

    success = generate_video_from_output(
        data_file,
        video_path=video_path,
        geometry=geometry,
        fps=args.fps,
        speedup=args.speedup,
        dpi=args.dpi,
    )

    # Clean up temporary merged file if it exists
    if data_file != output_file and data_file.exists():
        data_file.unlink()

    if success:
        logger.info("=" * 60)
        logger.info(f"✓ Video saved: {video_path}")
        logger.info("=" * 60)
    else:
        logger.error("✗ Video generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
