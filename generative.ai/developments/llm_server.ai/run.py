#!/usr/bin/env python3
"""LLM Server.AI — Local LLM Server with TUI Management"""

from src.ui.app import LLMServerApp


def main():
    app = LLMServerApp()
    app.run()


if __name__ == "__main__":
    main()
