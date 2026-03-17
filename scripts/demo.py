#!/usr/bin/env python3
"""
Interactive demo of FinBot sales agent.

Usage:
    python scripts/demo.py [OPTIONS]

This will start an interactive conversation with the AI sales agent.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.llm.agent import SalesAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print("   _____ _       ____        _   ")
    print("  |  ___(_)_ __ | __ )  ___ | |_ ")
    print("  | |_  | | '_ \\|  _ \\ / _ \\| __|")
    print("  |  _| | | | | | |_) | (_) | |_ ")
    print("  |_|   |_|_| |_|____/ \\___/ \\__|")
    print()
    print("  AI Sales Agent for Financial Products")
    print("=" * 70)
    print()


def print_help():
    """Print help message."""
    print("\nCommands:")
    print("  /help     - Show this help message")
    print("  /reset    - Reset conversation")
    print("  /score    - Show current lead score")
    print("  /summary  - Show conversation summary")
    print("  /quit     - Exit demo")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive demo of FinBot sales agent"
    )

    parser.add_argument(
        "--no-scoring",
        action="store_true",
        help="Disable ML lead scoring"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Override LLM model name"
    )

    args = parser.parse_args()

    print_banner()

    try:
        # Initialize agent
        print("Initializing FinBot agent...")
        agent = SalesAgent(
            use_scoring=not args.no_scoring,
            model_name=args.model
        )

        print("✓ Agent ready!\n")
        print("Type your message and press Enter to chat.")
        print("Type /help for commands, /quit to exit.\n")

        # Interactive loop
        while True:
            try:
                # Get user input
                user_input = input("\n\033[94mYou:\033[0m ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    command = user_input.lower()

                    if command == "/quit":
                        print("\nGoodbye! 👋")
                        break

                    elif command == "/help":
                        print_help()
                        continue

                    elif command == "/reset":
                        agent.reset_conversation()
                        print("\n✓ Conversation reset")
                        continue

                    elif command == "/score":
                        score = agent.current_lead_score
                        print(f"\n📊 Current Lead Score: {score:.3f} ({score*100:.1f}%)")
                        continue

                    elif command == "/summary":
                        summary = agent.get_conversation_summary()
                        print(f"\n📋 Conversation Summary:")
                        print(f"   Messages: {summary['message_count']}")
                        print(f"   User Messages: {summary['user_messages']}")
                        print(f"   Lead Score: {summary['current_lead_score']:.3f}")
                        continue

                    else:
                        print(f"\n❌ Unknown command: {command}")
                        print("Type /help for available commands")
                        continue

                # Process message
                result = agent.process_message(user_input)

                # Display response
                print(f"\n\033[92mFinBot:\033[0m {result['response']}")

                # Display metadata (in gray)
                print(f"\033[90m[Score: {result['lead_score_percentage']:.1f}% | "
                      f"Latency: {result['latency_ms']:.0f}ms", end="")

                if result['should_recommend']:
                    print(" | ⭐ High Interest", end="")

                print("]\033[0m")

                # Show recommendation if applicable
                if result['should_recommend'] and result['product_recommendation']:
                    print(f"\n\033[93m💡 Recommendation:\033[0m")
                    print(f"\033[90m{result['product_recommendation']}\033[0m")

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break

            except EOFError:
                print("\n\nGoodbye! 👋")
                break

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                print(f"\n❌ Error: {e}")
                print("Type /reset to reset the conversation or /quit to exit")

    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        print(f"\n❌ Failed to initialize agent: {e}")
        print("\nMake sure you have:")
        print("  1. Set your API key in .env (ANTHROPIC_API_KEY or OPENAI_API_KEY)")
        print("  2. Trained the ML model (python scripts/train.py)")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
