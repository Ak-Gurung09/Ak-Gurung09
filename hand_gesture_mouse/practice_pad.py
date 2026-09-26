"""
A harmless practice window for testing gestures in LIVE mode.

It counts left clicks, double clicks, right clicks, completed drags and scroll
position, so you can measure accuracy without risking clicks in real apps.

    python practice_pad.py      (in one terminal)
    python main.py              (in another terminal)

Needs tkinter: included with the python.org installers for Windows and macOS.
Linux: sudo apt install python3-tk. Homebrew Python: brew install python-tk.
"""

import sys
import tkinter as tk

LIST_LINES = 200
LOG_LINES = 6
SQUARE = 60  # size of the draggable square in pixels


class PracticePad:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Gesture Mouse Test Pad")
        self.counts = {"left": 0, "double": 0, "right": 0, "drags": 0}
        self.log_lines: list[str] = []

        top = tk.Frame(root, padx=10, pady=10)
        top.pack(fill="x")
        self.left_button = tk.Button(top, text="LEFT-CLICK ME", width=18, height=3, bg="#cde8ff",
                                     command=self.on_left)
        self.left_button.pack(side="left", padx=5)
        self.left_button.bind("<Double-Button-1>", self.on_double)

        right_area = tk.Label(top, text="RIGHT-CLICK HERE", width=18, height=3, bg="#ffe0c0", relief="ridge")
        right_area.pack(side="left", padx=5)
        # Tk on macOS has reported the right button as Button-2 or Button-3 depending on the version.
        right_buttons = ("<Button-2>", "<Button-3>") if sys.platform == "darwin" else ("<Button-3>",)
        for sequence in right_buttons:
            right_area.bind(sequence, self.on_right)

        tk.Button(top, text="Reset counters", command=self.reset).pack(side="left", padx=5)

        self.summary = tk.Label(root, font=("TkDefaultFont", 12, "bold"))
        self.summary.pack(anchor="w", padx=10)

        middle = tk.Frame(root, padx=10, pady=5)
        middle.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(middle, width=360, height=260, bg="white", highlightthickness=1)
        self.canvas.pack(side="left", padx=5)
        self.canvas.create_text(180, 15, text="Drag the blue square (pinch and hold)")
        self.square = self.canvas.create_rectangle(40, 100, 40 + SQUARE, 100 + SQUARE, fill="#3a7bd5")
        self.canvas.tag_bind(self.square, "<ButtonPress-1>", self.drag_begin)
        self.canvas.tag_bind(self.square, "<B1-Motion>", self.drag_move)
        self.canvas.tag_bind(self.square, "<ButtonRelease-1>", self.drag_end)
        self._drag_from = None

        list_frame = tk.Frame(middle)
        list_frame.pack(side="left", fill="both", expand=True, padx=5)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        self.listbox = tk.Listbox(list_frame, height=14, yscrollcommand=lambda a, b: self.on_scroll(scrollbar, a, b))
        for i in range(1, LIST_LINES + 1):
            self.listbox.insert("end", f"Scroll test line {i}")
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)
        self.scroll_label = tk.Label(root, text="Scroll the list with the two-finger gesture")
        self.scroll_label.pack(anchor="w", padx=10)

        self.log = tk.Label(root, justify="left", anchor="w", fg="#444")
        self.log.pack(fill="x", padx=10, pady=(0, 10))
        self.refresh()

    def record(self, text: str) -> None:
        self.log_lines = (self.log_lines + [text])[-LOG_LINES:]
        self.refresh()

    def refresh(self) -> None:
        c = self.counts
        self.summary.config(text=f"Left clicks: {c['left']}   Double clicks: {c['double']}   "
                                 f"Right clicks: {c['right']}   Drags: {c['drags']}")
        self.log.config(text="\n".join(self.log_lines) or "Events will appear here.")

    def on_left(self) -> None:
        self.counts["left"] += 1
        self.record("left click on the button")

    def on_double(self, _event) -> None:
        self.counts["double"] += 1
        self.record("double click")

    def on_right(self, _event) -> None:
        self.counts["right"] += 1
        self.record("right click")

    def drag_begin(self, event) -> None:
        self._drag_from = (event.x, event.y)

    def drag_move(self, event) -> None:
        if self._drag_from is None:
            return
        self.canvas.move(self.square, event.x - self._drag_from[0], event.y - self._drag_from[1])
        self._drag_from = (event.x, event.y)

    def drag_end(self, _event) -> None:
        if self._drag_from is not None:
            self.counts["drags"] += 1
            self.record("drag finished")
        self._drag_from = None

    def on_scroll(self, scrollbar, first, last) -> None:
        scrollbar.set(first, last)
        top_line = int(float(first) * LIST_LINES) + 1
        self.scroll_label.config(text=f"List scrolled to line {top_line}")

    def reset(self) -> None:
        self.counts = dict.fromkeys(self.counts, 0)
        self.log_lines = []
        self.refresh()


def main() -> None:
    root = tk.Tk()
    PracticePad(root)
    root.mainloop()


if __name__ == "__main__":
    main()
