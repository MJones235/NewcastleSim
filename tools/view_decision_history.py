#!/usr/bin/env python3
"""
Decision history viewer for Station Concordia simulations.

Displays agent decision logs from completed simulation runs with:
- Timeline view of all decisions
- Filtering by agent, time, or action type
- Search functionality
- Export capabilities

Usage:
    python tools/view_decision_history.py --output-file PATH
    python tools/view_decision_history.py --output-file scenarios/station_concordia/output/run_20240210_120000/agent_decisions.json
"""

import argparse
import json
import sys
from pathlib import Path

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Warning: tkinter not available. Install with system package manager.")


class DecisionHistoryViewer:
    """GUI viewer for agent decision history."""

    def __init__(self, output_file: Path):
        """Initialize the decision history viewer."""
        self.output_file = output_file
        self.data = self._load_data()

        if not self.data:
            raise ValueError(f"Could not load data from {output_file}")

        # Extract decision timeline
        self.decisions_timeline = self._build_timeline()
        self.filtered_decisions = self.decisions_timeline.copy()

        # Setup GUI
        self.root = tk.Tk()
        self.root.title(f"Decision History: {output_file.parent.name}")
        self.root.geometry("1200x800")

        self._setup_ui()
        self._populate_initial_view()

    def _load_data(self) -> dict:
        """Load simulation output data."""
        try:
            with open(self.output_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load data: {e}")
            return {}

    def _build_timeline(self) -> list:
        """
        Build chronological timeline of all decisions.

        Returns:
            List of decision dicts with keys: time, agent_id, action, observation, translated
        """
        timeline = []
        agent_decisions = self.data.get("agent_decisions", {})

        for agent_id, agent_data in agent_decisions.items():
            if isinstance(agent_data, dict) and "decisions" in agent_data:
                for decision in agent_data["decisions"]:
                    timeline.append(
                        {
                            "agent_id": agent_id,
                            "time": decision.get("time", decision.get("timestamp", 0)),
                            "action": decision.get("action", "Unknown"),
                            "observation": decision.get("observation", ""),
                            "translated": decision.get("translated", {}),
                            "reasoning": decision.get("reasoning", ""),
                        }
                    )

        # Sort by time
        timeline.sort(key=lambda x: x["time"])

        return timeline

    def _setup_ui(self):
        """Setup the user interface."""
        # Top frame: Info and filters
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)

        # Simulation info
        info_text = (
            f"Simulation Time: {self.data.get('final_time', self.data.get('current_time', 0)):.1f}s | "
            f"Total Decisions: {len(self.decisions_timeline)} | "
            f"Agents: {len(self.data.get('agent_decisions', {}))}"
        )
        info_label = ttk.Label(top_frame, text=info_text, font=("Arial", 10, "bold"))
        info_label.pack(side=tk.LEFT)

        # Filter frame
        filter_frame = ttk.LabelFrame(self.root, text="Filters", padding="10")
        filter_frame.pack(fill=tk.X, padx=10, pady=5)

        # Agent filter
        ttk.Label(filter_frame, text="Agent:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.agent_filter = ttk.Combobox(filter_frame, width=15)
        self.agent_filter["values"] = ["All"] + sorted(self.data.get("agent_decisions", {}).keys())
        self.agent_filter.set("All")
        self.agent_filter.grid(row=0, column=1, padx=5)
        self.agent_filter.bind("<<ComboboxSelected>>", lambda e: self._apply_filters())

        # Time range filter
        ttk.Label(filter_frame, text="Time Range:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.time_start = ttk.Entry(filter_frame, width=10)
        self.time_start.insert(0, "0")
        self.time_start.grid(row=0, column=3, padx=5)
        ttk.Label(filter_frame, text="to").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.time_end = ttk.Entry(filter_frame, width=10)
        max_time = self.data.get("final_time", self.data.get("current_time", 999))
        self.time_end.insert(0, str(max_time))
        self.time_end.grid(row=0, column=5, padx=5)

        # Action type filter
        ttk.Label(filter_frame, text="Action Type:").grid(row=0, column=6, sticky=tk.W, padx=5)
        self.action_filter = ttk.Combobox(filter_frame, width=15)
        action_types = set()
        for dec in self.decisions_timeline:
            if dec.get("translated", {}).get("action_type"):
                action_types.add(dec["translated"]["action_type"])
        self.action_filter["values"] = ["All"] + sorted(action_types)
        self.action_filter.set("All")
        self.action_filter.grid(row=0, column=7, padx=5)
        self.action_filter.bind("<<ComboboxSelected>>", lambda e: self._apply_filters())

        # Search box
        ttk.Label(filter_frame, text="Search:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.search_box = ttk.Entry(filter_frame, width=40)
        self.search_box.grid(row=1, column=1, columnspan=4, padx=5, pady=5, sticky=tk.EW)
        self.search_box.bind("<KeyRelease>", lambda e: self._apply_filters())

        # Apply button
        ttk.Button(filter_frame, text="Apply Filters", command=self._apply_filters).grid(
            row=1, column=5, columnspan=2, padx=5, pady=5
        )

        # Reset button
        ttk.Button(filter_frame, text="Reset", command=self._reset_filters).grid(
            row=1, column=7, padx=5, pady=5
        )

        # Main content frame with treeview
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create treeview for decisions list
        columns = ("Time", "Agent", "Action", "Type")
        self.tree = ttk.Treeview(main_frame, columns=columns, show="tree headings", height=15)

        # Configure columns
        self.tree.heading("#0", text="ID")
        self.tree.column("#0", width=50)
        self.tree.heading("Time", text="Time (s)")
        self.tree.column("Time", width=80)
        self.tree.heading("Agent", text="Agent")
        self.tree.column("Agent", width=100)
        self.tree.heading("Action", text="Action")
        self.tree.column("Action", width=400)
        self.tree.heading("Type", text="Type")
        self.tree.column("Type", width=100)

        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self._on_decision_select)

        # Detail frame
        detail_frame = ttk.LabelFrame(self.root, text="Decision Details", padding="10")
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Detail text area
        self.detail_text = scrolledtext.ScrolledText(detail_frame, height=10, wrap=tk.WORD)
        self.detail_text.pack(fill=tk.BOTH, expand=True)

        # Bottom frame: Statistics and export
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(fill=tk.X)

        self.status_label = ttk.Label(bottom_frame, text="", font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT)

        ttk.Button(bottom_frame, text="Export Filtered", command=self._export_filtered).pack(
            side=tk.RIGHT, padx=5
        )
        ttk.Button(bottom_frame, text="Statistics", command=self._show_statistics).pack(
            side=tk.RIGHT, padx=5
        )

    def _populate_initial_view(self):
        """Populate the treeview with decisions."""
        self._update_tree_view()

    def _update_tree_view(self):
        """Update the treeview with filtered decisions."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Add filtered decisions
        for idx, decision in enumerate(self.filtered_decisions):
            action_short = (
                decision["action"][:80] + "..."
                if len(decision["action"]) > 80
                else decision["action"]
            )
            action_type = decision.get("translated", {}).get("action_type", "unknown")

            self.tree.insert(
                "",
                tk.END,
                text=str(idx + 1),
                values=(f"{decision['time']:.1f}", decision["agent_id"], action_short, action_type),
            )

        # Update status
        self.status_label.config(
            text=f"Showing {len(self.filtered_decisions)} of {len(self.decisions_timeline)} decisions"
        )

    def _apply_filters(self):
        """Apply current filters to decision list."""
        filtered = []

        # Get filter values
        agent_filter = self.agent_filter.get()
        try:
            time_start = float(self.time_start.get())
        except ValueError:
            time_start = 0
        try:
            time_end = float(self.time_end.get())
        except ValueError:
            time_end = float("inf")
        action_filter = self.action_filter.get()
        search_text = self.search_box.get().lower()

        # Apply filters
        for decision in self.decisions_timeline:
            # Agent filter
            if agent_filter != "All" and decision["agent_id"] != agent_filter:
                continue

            # Time range filter
            if not (time_start <= decision["time"] <= time_end):
                continue

            # Action type filter
            if action_filter != "All":
                if decision.get("translated", {}).get("action_type") != action_filter:
                    continue

            # Search filter (searches in action and observation)
            if search_text:
                searchable = (
                    decision["action"].lower()
                    + " "
                    + decision.get("observation", "").lower()
                    + " "
                    + decision.get("reasoning", "").lower()
                )
                if search_text not in searchable:
                    continue

            filtered.append(decision)

        self.filtered_decisions = filtered
        self._update_tree_view()

    def _reset_filters(self):
        """Reset all filters to default."""
        self.agent_filter.set("All")
        self.time_start.delete(0, tk.END)
        self.time_start.insert(0, "0")
        max_time = self.data.get("final_time", self.data.get("current_time", 999))
        self.time_end.delete(0, tk.END)
        self.time_end.insert(0, str(max_time))
        self.action_filter.set("All")
        self.search_box.delete(0, tk.END)
        self._apply_filters()

    def _on_decision_select(self, event):
        """Handle decision selection in treeview."""
        selection = self.tree.selection()
        if not selection:
            return

        # Get selected item index
        item = self.tree.item(selection[0])
        idx = int(item["text"]) - 1

        if 0 <= idx < len(self.filtered_decisions):
            decision = self.filtered_decisions[idx]
            self._display_decision_details(decision)

    def _display_decision_details(self, decision: dict):
        """Display full details of a decision."""
        self.detail_text.delete(1.0, tk.END)

        details = f"""
Time: {decision['time']:.2f}s
Agent: {decision['agent_id']}

ACTION:
{decision['action']}

OBSERVATION:
{decision.get('observation', 'N/A')}

REASONING:
{decision.get('reasoning', 'N/A')}

TRANSLATED ACTION:
  Type: {decision.get('translated', {}).get('action_type', 'N/A')}
  Destination: {decision.get('translated', {}).get('destination', 'N/A')}
  Target: {decision.get('translated', {}).get('target', 'N/A')}
  Wait Reason: {decision.get('translated', {}).get('wait_reason', 'N/A')}
        """

        self.detail_text.insert(1.0, details.strip())

    def _show_statistics(self):
        """Show statistics about the decision history."""
        # Calculate statistics
        total_decisions = len(self.decisions_timeline)
        agents = {d["agent_id"] for d in self.decisions_timeline}

        action_types = {}
        for decision in self.decisions_timeline:
            action_type = decision.get("translated", {}).get("action_type", "unknown")
            action_types[action_type] = action_types.get(action_type, 0) + 1

        decisions_per_agent = {}
        for decision in self.decisions_timeline:
            agent = decision["agent_id"]
            decisions_per_agent[agent] = decisions_per_agent.get(agent, 0) + 1

        # Format statistics
        stats = f"""
SIMULATION STATISTICS

Total Decisions: {total_decisions}
Number of Agents: {len(agents)}
Simulation Time: {self.data.get('final_time', self.data.get('current_time', 0)):.1f}s

DECISIONS BY ACTION TYPE:
"""
        for action_type, count in sorted(action_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_decisions * 100) if total_decisions > 0 else 0
            stats += f"  {action_type}: {count} ({percentage:.1f}%)\n"

        stats += "\nDECISIONS PER AGENT:\n"
        stats += f"  Average: {total_decisions / len(agents):.1f}\n"
        stats += f"  Max: {max(decisions_per_agent.values())}\n"
        stats += f"  Min: {min(decisions_per_agent.values())}\n"

        # Show in message box
        messagebox.showinfo("Statistics", stats)

    def _export_filtered(self):
        """Export filtered decisions to JSON file."""
        from tkinter import filedialog

        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="filtered_decisions.json",
        )

        if not filename:
            return

        # Export filtered decisions
        export_data = {
            "source_file": str(self.output_file),
            "filter_applied": {
                "agent": self.agent_filter.get(),
                "time_start": self.time_start.get(),
                "time_end": self.time_end.get(),
                "action_type": self.action_filter.get(),
                "search": self.search_box.get(),
            },
            "total_decisions": len(self.filtered_decisions),
            "decisions": self.filtered_decisions,
        }

        try:
            with open(filename, "w") as f:
                json.dump(export_data, f, indent=2)
            messagebox.showinfo(
                "Success", f"Exported {len(self.filtered_decisions)} decisions to {filename}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def main():
    """Main entry point."""
    if not TKINTER_AVAILABLE:
        print("Error: tkinter is required for the GUI viewer")
        print("Install with: sudo apt-get install python3-tk (Ubuntu/Debian)")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="View agent decision history from Station Concordia simulation"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to agent decisions JSON file",
    )
    args = parser.parse_args()

    output_file = Path(args.output_file)
    if not output_file.exists():
        print(f"Error: Output file not found: {output_file}")
        sys.exit(1)

    try:
        viewer = DecisionHistoryViewer(output_file)
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
