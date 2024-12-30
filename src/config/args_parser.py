import argparse

from .config import Lang

mapio_parser = argparse.ArgumentParser(description="MapIO, with LLM integration")

mapio_parser.add_argument("--model", help="Path to model json file.", required=True)
mapio_parser.add_argument(
    "--out",
    help="Path to chat save file.",
    default="out/last_chat.txt",
)

mapio_parser.add_argument(
    "--lang", help="System language", type=Lang, choices=list(Lang), default=Lang.EN
)
mapio_parser.add_argument(
    "--tts-rate",
    help="TTS speed rate (words per minute).",
    type=int,
    default=200,
)

mapio_parser.add_argument(
    "--no-llm",
    help="Disable llm interaction.",
    action="store_true",
    default=False,
)
mapio_parser.add_argument(
    "--no-stt",
    help="Replace STT with keyboard input.",
    action="store_true",
    default=False,
)

mapio_parser.add_argument(
    "--debug",
    help="Enable debug mode.",
    action="store_true",
    default=False,
)

get_args = mapio_parser.parse_args
