# Live Concordia Viewer

Real-time viewer for Station Concordia simulation progress.

## Features

- **Live Updates**: Displays decisions as they happen during simulation
- **Parsed LLM Responses**: Shows each question/answer pair from Concordia's reasoning
- **Game Master Actions**: Displays translated actions and targets
- **Color Formatting**: Uses rich library for beautiful terminal output (optional)

## Installation

```bash
# Optional: Install rich for better formatting
pip install rich
```

## Usage

### Start the Viewer (in one terminal)
```bash
python tools/view_concordia_live.py
```

### Run the Simulation (in another terminal)
```bash
python scenarios/station_concordia/run_station_concordia.py
```

The viewer will automatically detect and display new decisions as they are made.

## Display Format

For each decision, you'll see:

### LLM REASONING
- **Self**: Agent's personality and characteristics
- **Situation**: Current context and environment
- **Risk**: Danger level assessment
- **Social**: What other people are doing
- **Strategy**: What a person like this would do
- **Action**: Final decision on what to do next

### GAME MASTER
- **Type**: Action type (move, wait, help, etc.)
- **Target**: Coordinates or target location
- **Confidence**: Translation confidence (0-100%)
- **Reasoning**: Why this translation was chosen

## Options

```bash
# Specify custom output file
python tools/view_concordia_live.py --output-file path/to/decisions.json
```

## Example Output

```
┌─────────────────────────── agent_0 ───────────────────────────┐
│ Time: 5.0s                                                     │
│                                                                │
│ ═══ LLM REASONING ═══                                         │
│ Self: Agent 0 is a 35-year-old ISTJ: practical, fact-minded  │
│ Situation: Emergency evacuation, 50m from north_exit          │
│ Risk: Danger level perceived as low                           │
│ Social: The area is empty; no other people nearby            │
│ Strategy: Follow announcements and evacuate immediately       │
│ Action: Move toward north_exit at brisk, steady pace         │
│                                                                │
│ ═══ GAME MASTER ═══                                           │
│ Type: move                                                     │
│ Target: [50.0, 100.0]                                         │
│ Confidence: 90.0%                                             │
│ Reasoning: Moving to north_exit exit                          │
└────────────────────────────────────────────────────────────────┘
```

## Notes

- Updates every 2 seconds
- Works even without `rich` library (falls back to simple text)
- Can be started before or after the simulation
- Press Ctrl+C to stop
