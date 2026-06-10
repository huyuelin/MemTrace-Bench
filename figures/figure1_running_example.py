#!/usr/bin/env python3
"""
Generate Figure 1: Running example of LLM call sequence.

This script creates a visual diagram showing a complete agent execution flow,
including LLM calls, tool calls, and memory operations. The diagram illustrates
how persistent memory can become a hidden dependency in stateful coding agents.

The figure shows:
- Sequential steps in the agent execution
- LLM invocation boxes with prompts and responses
- Tool call operations (read file, edit file, run tests, etc.)
- Memory store/retrieve operations
- Flow arrows connecting the steps

Usage:
    python figure1_running_example.py --output data/results/figures/figure1.pdf
    python figure1_running_example.py --use-mock --sequence-id mt-v5-py-auth-0017
"""

import argparse
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Color scheme for different element types
COLORS = {
    "llm_call": "#E3F2FD",        # Light blue for LLM calls
    "tool_call": "#E8F5E9",       # Light green for tool calls
    "memory_store": "#FFF3E0",     # Light orange for memory store
    "memory_retrieve": "#F3E5F5", # Light purple for memory retrieve
    "arrow": "#424242",            # Dark gray for arrows
    "border": "#1976D2",          # Blue border for LLM calls
    "text": "#212121",             # Dark text
    "background": "#FFFFFF",       # White background
}


def create_mock_sequence():
    """
    Create a mock running example sequence for Figure 1.

    This sequence illustrates a typical agent execution flow where:
    1. Agent receives a task prompt
    2. Agent calls LLM to understand the task
    3. Agent reads files to understand the codebase
    4. Agent calls LLM again with context
    5. Agent makes edits to fix the issue
    6. Agent runs tests to verify the fix
    7. Agent stores memory for future tasks
    8. Agent retrieves memory in a later task (showing hidden dependency)

    Returns:
        list: A list of step dictionaries, each containing:
            - step_num: Step number in the sequence
            - step_type: Type of step ("llm_call", "tool_call", "memory_store", "memory_retrieve")
            - title: Short title for the step
            - content: Detailed content/description
            - inputs: List of input descriptions
            - outputs: List of output descriptions
    """
    sequence = [
        {
            "step_num": 1,
            "step_type": "llm_call",
            "title": "Step 1: Task Understanding",
            "content": "LLM receives the task prompt and understands the requirement",
            "inputs": ["Task: Fix token verification in authlib 0.14"],
            "outputs": ["Plan: Need to check authlib version and token verifier implementation"]
        },
        {
            "step_num": 2,
            "step_type": "tool_call",
            "title": "Step 2: Read File",
            "content": "Agent reads the authentication module to understand current implementation",
            "inputs": ["File path: src/auth/token.py"],
            "outputs": ["File content: class TokenVerifier with legacy verifier method"]
        },
        {
            "step_num": 3,
            "step_type": "llm_call",
            "title": "Step 3: Analysis",
            "content": "LLM analyzes the code and identifies the issue with legacy verifier",
            "inputs": ["Code context from token.py", "Task: Fix token verification"],
            "outputs": ["Analysis: Legacy verifier uses 60-second leeway which is insecure"]
        },
        {
            "step_num": 4,
            "step_type": "tool_call",
            "title": "Step 4: Edit File",
            "content": "Agent edits the token verifier to remove legacy leeway parameter",
            "inputs": ["Edit: Remove leeway=60 from verifier"],
            "outputs": ["File edited successfully"]
        },
        {
            "step_num": 5,
            "step_type": "tool_call",
            "title": "Step 5: Run Tests",
            "content": "Agent runs tests to verify the fix works correctly",
            "inputs": ["Test command: pytest tests/auth/"],
            "outputs": ["Tests pass: 15 passed, 0 failed"]
        },
        {
            "step_num": 6,
            "step_type": "memory_store",
            "title": "Step 6: Store Memory",
            "content": "Agent stores memory: use legacy verifier with 60-second leeway (for authlib 0.14)",
            "inputs": ["Memory content: use legacy verifier with 60-second leeway"],
            "outputs": ["Memory stored with scope: authlib 0.14, tag: security-sensitive"]
        },
        {
            "step_num": 7,
            "step_type": "memory_retrieve",
            "title": "Step 7: Retrieve Memory (Hidden Dependency)",
            "content": "Later task: Agent retrieves memory without checking scope - authlib now 1.2!",
            "inputs": ["Query: token verification approach"],
            "outputs": ["Retrieved memory: use legacy verifier with 60-second leeway"]
        },
        {
            "step_num": 8,
            "step_type": "llm_call",
            "title": "Step 8: Wrong Action (Harm)",
            "content": "LLM uses outdated memory, adds legacy verifier to authlib 1.2 code",
            "inputs": ["Retrieved memory: use legacy verifier", "Task: Fix token verification authlib 1.2"],
            "outputs": ["Patch: Added legacy verifier with leeway=60 (WRONG for authlib 1.2!)"]
        },
    ]
    return sequence


def draw_step(ax, step, x, y, box_width, box_height):
    """
    Draw a single step box with its inputs and outputs.

    Args:
        ax: Matplotlib axes to draw on
        step: Step dictionary containing step information
        x: X coordinate of the box center
        y: Y coordinate of the box center
        box_width: Width of the box
        box_height: Height of the box

    Returns:
        tuple: (x, y) coordinates of the box center (for arrow drawing)
    """
    step_type = step["step_type"]
    color = COLORS.get(step_type, COLORS["llm_call"])
    border_color = COLORS["border"] if step_type == "llm_call" else COLORS["arrow"]

    # Draw the main box
    box = FancyBboxPatch(
        (x - box_width/2, y - box_height/2),
        box_width, box_height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor=border_color,
        linewidth=2,
        zorder=3
    )
    ax.add_patch(box)

    # Draw step number circle
    circle = mpatches.Circle((x - box_width/2 + 0.15, y + box_height/2 - 0.15), 0.12,
                       color=border_color, zorder=4)
    ax.add_patch(circle)
    ax.text(x - box_width/2 + 0.15, y + box_height/2 - 0.15, str(step["step_num"]),
             ha="center", va="center", fontsize=8, fontweight="bold",
             color="white", zorder=5)

    # Draw title
    ax.text(x, y + box_height/4, step["title"],
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=COLORS["text"], zorder=5, wrap=True)

    # Draw content (truncated)
    content_lines = step["content"].split(", ")[:2]  # Limit to 2 lines
    content_y = y - box_height/4
    for i, line in enumerate(content_lines):
        if len(line) > 40:
            line = line[:37] + "..."
        ax.text(x, content_y - i*0.15, line,
                ha="center", va="center", fontsize=7,
                color=COLORS["text"], zorder=5, style="italic")

    return x, y


def draw_arrow(ax, x1, y1, x2, y2, style="straight"):
    """
    Draw an arrow between two points.

    Args:
        ax: Matplotlib axes to draw on
        x1, y1: Start coordinates
        x2, y2: End coordinates
        style: Arrow style ("straight" or "curved")
    """
    if style == "curved":
        # Draw curved arrow using ConnectionPatch
        arrow = mpatches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->,head_width=0.4,head_length=0.8",
            color=COLORS["arrow"],
            linewidth=2,
            connectionstyle="arc3,rad=0.3",
            zorder=2
        )
    else:
        # Draw straight arrow
        arrow = mpatches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->,head_width=0.4,head_length=0.8",
            color=COLORS["arrow"],
            linewidth=2,
            zorder=2
        )
    ax.add_patch(arrow)


def generate_figure1(sequence, output_path):
    """
    Generate Figure 1: Running example LLM call sequence diagram.

    Args:
        sequence: List of step dictionaries
        output_path: Path to save the PDF file
    """
    # Create figure with appropriate size
    fig_width = 16
    fig_height = 10
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Set plot limits and remove axes
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis("off")

    # Title
    fig.suptitle("Figure 1: Running Example of LLM Call Sequence in Stateful Coding Agent",
                 fontsize=14, fontweight="bold", y=0.98)

    # Subtitle explaining the hidden dependency
    ax.text(fig_width/2, fig_height - 0.6,
            "Memory stored in Task 1 becomes a hidden dependency in Task 2, causing harmful behavior",
            ha="center", va="center", fontsize=10, style="italic", color="red")

    # Layout parameters
    n_steps = len(sequence)
    box_width = 1.5
    box_height = 1.8
    x_start = 1.0
    x_spacing = (fig_width - 2 - box_width) / (n_steps - 1) if n_steps > 1 else 0

    # Draw steps
    step_positions = []
    for i, step in enumerate(sequence):
        x = x_start + i * x_spacing
        y = fig_height / 2

        # Adjust y position for memory operations (draw them slightly offset)
        if step["step_type"] == "memory_store":
            y = fig_height / 2 + 2.0
        elif step["step_type"] == "memory_retrieve":
            y = fig_height / 2 - 2.0

        pos_x, pos_y = draw_step(ax, step, x, y, box_width, box_height)
        step_positions.append((pos_x, pos_y, step["step_type"], y))

    # Draw arrows between steps
    for i in range(len(step_positions) - 1):
        x1, y1, type1, orig_y1 = step_positions[i]
        x2, y2, type2, orig_y2 = step_positions[i + 1]

        # Determine arrow style
        if type1 == "memory_store" and type2 == "memory_retrieve":
            # Memory store to retrieve - show as hidden dependency (dashed, curved)
            draw_arrow(ax, x1 + box_width/2, orig_y1, x2 - box_width/2, orig_y2, style="curved")
        elif type1 == "memory_retrieve":
            # Memory retrieve to LLM call - solid arrow
            draw_arrow(ax, x1 + box_width/2, orig_y1, x2 - box_width/2, orig_y2)
        else:
            # Normal sequential flow
            draw_arrow(ax, x1 + box_width/2, y1, x2 - box_width/2, y2)

    # Add annotations for key concepts
    annotation_y_start = 0.8
    annotations = [
        "Prelude Task: Agent fixes issue, stores memory",
        "Probe Task: Agent retrieves memory, causes harm",
    ]

    # Color-coded boxes for prelude vs probe
    prelude_box = mpatches.FancyBboxPatch(
        (0.5, annotation_y_start - 0.1), 7.0, 0.5,
        boxstyle="round,pad=0.05",
        facecolor="#C8E6C9", edgecolor="green", linewidth=2, zorder=1
    )
    ax.add_patch(prelude_box)
    ax.text(4.0, annotation_y_start + 0.15, annotations[0],
            ha="center", va="center", fontsize=9, fontweight="bold")

    probe_box = mpatches.FancyBboxPatch(
        (8.5, annotation_y_start - 0.1), 7.0, 0.5,
        boxstyle="round,pad=0.05",
        facecolor="#FFCDD2", edgecolor="red", linewidth=2, zorder=1
    )
    ax.add_patch(probe_box)
    ax.text(12.0, annotation_y_start + 0.15, annotations[1],
            ha="center", va="center", fontsize=9, fontweight="bold")

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["llm_call"], edgecolor=COLORS["border"],
                       label="LLM Call"),
        mpatches.Patch(facecolor=COLORS["tool_call"], edgecolor=COLORS["arrow"],
                       label="Tool Call"),
        mpatches.Patch(facecolor=COLORS["memory_store"], edgecolor=COLORS["arrow"],
                       label="Memory Store"),
        mpatches.Patch(facecolor=COLORS["memory_retrieve"], edgecolor=COLORS["arrow"],
                       label="Memory Retrieve"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=10,
              frameon=True, fancybox=True, shadow=True)

    # Add note about the hidden dependency
    ax.text(fig_width/2, 0.3,
            "Note: Memory retrieved in Step 7 is outdated (authlib 0.14 -> 1.2), causing wrong action in Step 8",
            ha="center", va="center", fontsize=9, color="red", style="italic")

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as PDF
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure 1 saved to {output_path}")

    # Close the figure to free memory
    plt.close(fig)


def load_sequence_from_file(sequence_id, data_dir):
    """
    Load a sequence from JSON files in the data directory.

    Args:
        sequence_id: ID of the sequence to load
        data_dir: Directory containing sequence JSON files

    Returns:
        list: Sequence data if found, None otherwise

    Raises:
        AssertionError: If data_dir doesn't exist or sequence_id is invalid
    """
    # Fast-Fail: Check that data_dir exists
    assert os.path.isdir(data_dir), f"Data directory not found: {data_dir}"

    # Fast-Fail: Check that sequence_id is a non-empty string
    assert isinstance(sequence_id, str) and len(sequence_id) > 0, \
        f"Invalid sequence_id: {sequence_id}"

    # Look for sequence file
    sequence_file = os.path.join(data_dir, f"sequence_{sequence_id}.json")

    # Fast-Fail: Check that sequence file exists
    assert os.path.isfile(sequence_file), f"Sequence file not found: {sequence_file}"

    # Load and return sequence data
    with open(sequence_file, "r") as f:
        sequence_data = json.load(f)

    # Fast-Fail: Check that sequence_data is a list
    assert isinstance(sequence_data, list), \
        f"Sequence data should be a list, got {type(sequence_data)}"

    # Fast-Fail: Check that sequence has at least one step
    assert len(sequence_data) > 0, f"Sequence {sequence_id} has no steps"

    return sequence_data


def main():
    """Main function to parse arguments and generate Figure 1."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 1: Running example LLM call sequence diagram"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/figures/figure1.pdf",
        help="Output PDF file path (default: data/results/figures/figure1.pdf)"
    )
    parser.add_argument(
        "--sequence-id",
        type=str,
        default=None,
        help="Sequence ID to visualize (default: use mock data)"
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock data instead of loading from file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed sequence data (default: data/processed)"
    )

    args = parser.parse_args()

    # Fast-Fail: Check output path is valid
    output_dir = os.path.dirname(args.output)
    if output_dir:  # Only check if output_dir is not empty
        assert output_dir == "" or os.path.isdir(output_dir) or not os.path.exists(output_dir), \
            f"Output directory path is invalid: {output_dir}"

    # Load or create sequence data
    if args.use_mock or args.sequence_id is None:
        print("Using mock data for Figure 1")
        sequence = create_mock_sequence()
    else:
        print(f"Loading sequence {args.sequence_id} from {args.data_dir}")
        sequence = load_sequence_from_file(args.sequence_id, args.data_dir)

    # Fast-Fail: Check that sequence is valid
    assert sequence is not None and len(sequence) > 0, \
        "Sequence data is empty or None"

    # Generate the figure
    generate_figure1(sequence, args.output)

    print("Figure 1 generation complete!")


if __name__ == "__main__":
    main()
