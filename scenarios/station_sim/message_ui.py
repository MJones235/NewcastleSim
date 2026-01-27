"""
UI interface for sending messages to agents during simulation.
"""

import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import Optional


class MessageUI:
    """
    Simple UI window for sending messages to all agents in the simulation.
    """

    def __init__(self, on_message_callback: Callable[[str], None]):
        """
        Initialize the message UI.

        Args:
            on_message_callback: Function to call when a message is submitted
        """
        self.on_message_callback = on_message_callback
        self.window: Optional[tk.Tk] = None
        self.message_entry: Optional[tk.Entry] = None
        self.message_history: Optional[tk.Text] = None

    def create_window(self):
        """Create and configure the UI window"""
        self.window = tk.Tk()
        self.window.title("Broadcast Message to Agents")
        self.window.geometry("500x400")

        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Title label
        title_label = ttk.Label(
            main_frame, text="Send Message to All Agents", font=("Arial", 14, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Message history (read-only text area)
        history_label = ttk.Label(main_frame, text="Message History:")
        history_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        self.message_history = tk.Text(
            main_frame, height=15, width=50, state="disabled", wrap=tk.WORD
        )
        self.message_history.grid(
            row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 10)
        )

        # Scrollbar for history
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.message_history.yview)
        scrollbar.grid(row=2, column=2, sticky=(tk.N, tk.S))
        self.message_history["yscrollcommand"] = scrollbar.set

        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        input_frame.columnconfigure(0, weight=1)

        # Message entry label
        entry_label = ttk.Label(input_frame, text="New Message:")
        entry_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        # Message entry field
        self.message_entry = ttk.Entry(input_frame, width=40)
        self.message_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.message_entry.bind("<Return>", lambda e: self.submit_message())

        # Submit button
        submit_button = ttk.Button(input_frame, text="Send", command=self.submit_message)
        submit_button.grid(row=1, column=1)

        # Focus on entry field
        self.message_entry.focus()

    def submit_message(self):
        """Handle message submission"""
        message = self.message_entry.get().strip()
        if message:
            # Call the callback to broadcast to agents
            self.on_message_callback(message)

            # Add to history
            self.add_to_history(message)

            # Clear entry field
            self.message_entry.delete(0, tk.END)

    def add_to_history(self, message: str):
        """Add a message to the history display"""
        self.message_history.config(state="normal")
        self.message_history.insert(tk.END, f"{message}\n")
        self.message_history.see(tk.END)  # Auto-scroll to bottom
        self.message_history.config(state="disabled")

    def run(self):
        """Start the UI in a separate thread"""

        def start_ui():
            self.create_window()
            self.window.mainloop()

        ui_thread = threading.Thread(target=start_ui, daemon=True)
        ui_thread.start()

    def close(self):
        """Close the UI window"""
        if self.window:
            self.window.quit()
            self.window.destroy()
