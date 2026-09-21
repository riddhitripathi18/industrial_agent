"""
Industrial Plant Monitoring Agent — CLI Entry Point
====================================================
Interactive terminal interface for the agent.

Usage:
    python -m agent.run

    Or with a single question (non-interactive):
    python -m agent.run "How is E02 doing?"

Commands during interactive session:
    /reset      Start a fresh conversation
    /tools      List all available tools
    /history    Show conversation turn count
    /quit       Exit
"""

import sys
import os
import argparse
import textwrap

# Ensure project root is in path when running as __main__
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core import IndustrialAgent, AGENT_TOOLS

# ─────────────────────────────────────────────
# Terminal colours (Windows-safe)
# ─────────────────────────────────────────────
try:
    import colorama
    colorama.init()
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
except ImportError:
    GREEN = CYAN = YELLOW = RED = BOLD = RESET = ""

BANNER = f"""{BOLD}{CYAN}
╔══════════════════════════════════════════════════════════════╗
║     IPMA — Industrial Plant Monitoring Agent                 ║
║     Powered by Gemini · Raw SDK · Phase 4                    ║
╚══════════════════════════════════════════════════════════════╝
{RESET}"""

HINT = f"""{YELLOW}
Equipment available:
  Heat Exchangers : E01, E02, E03, E04, E05
  Bearing Tests   : Test 1 (Bearing1-4 x/y), Test 2 (Bearing1-4), Test 3 (Bearing1-4)

Example questions:
  → "Give me a full fleet overview"
  → "Is E02 fouling? What is the energy loss?"
  → "Which bearing in Test 2 is closest to failure?"
  → "What does ISO Zone C mean and what action should I take?"
  → "Walk me through the chemical cleaning procedure for E01"

Commands: /reset  /tools  /history  /quit
{RESET}"""


def print_response(text: str, width: int = 100) -> None:
    """Pretty-print agent response with word wrap."""
    print()
    for paragraph in text.split("\n"):
        if paragraph.strip():
            wrapped = textwrap.fill(paragraph, width=width)
            print(f"{GREEN}{wrapped}{RESET}")
        else:
            print()
    print()


def handle_command(cmd: str, agent: IndustrialAgent) -> bool:
    """Handle slash commands. Returns True to continue, False to quit."""
    cmd = cmd.strip().lower()
    if cmd in ("/quit", "/exit", "/q"):
        print(f"\n{CYAN}Goodbye. Stay safe out there.{RESET}\n")
        return False
    elif cmd == "/reset":
        agent.reset()
        print(f"{YELLOW}Conversation reset.{RESET}")
    elif cmd == "/tools":
        print(f"\n{CYAN}Registered tools ({len(AGENT_TOOLS)}):{RESET}")
        for fn in AGENT_TOOLS:
            first_line = (fn.__doc__ or "").strip().split("\n")[0]
            print(f"  {BOLD}{fn.__name__}{RESET} — {first_line}")
        print()
    elif cmd == "/history":
        turns = len(agent.history) // 2
        print(f"{CYAN}{turns} conversation turn(s) in current session.{RESET}")
    else:
        print(f"{YELLOW}Unknown command: {cmd}. Try /reset, /tools, /history, /quit{RESET}")
    return True


def run_interactive(agent: IndustrialAgent) -> None:
    """Run the interactive chat loop."""
    print(BANNER)
    print(HINT)

    while True:
        try:
            user_input = input(f"{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{CYAN}Interrupted. Goodbye.{RESET}\n")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not handle_command(user_input, agent):
                break
            continue

        print(f"\n{CYAN}Agent is thinking...{RESET}")
        response = agent.ask(user_input)
        print(f"{BOLD}Agent:{RESET}")
        print_response(response)


def run_single(agent: IndustrialAgent, question: str) -> None:
    """Answer a single question and exit (non-interactive mode)."""
    print(f"\n{CYAN}Processing: {question}{RESET}\n")
    response = agent.ask(question)
    print_response(response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IPMA — Industrial Plant Monitoring Agent"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Single question (non-interactive mode). Leave blank for interactive chat.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model name (default: from .env or gemini-2.0-flash)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    # Initialize agent (will raise if GOOGLE_API_KEY not set)
    try:
        agent = IndustrialAgent(model_name=args.model, verbose=args.verbose)
    except ValueError as e:
        print(f"\n{RED}Configuration error:{RESET} {e}\n")
        sys.exit(1)

    if args.question:
        run_single(agent, args.question)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
