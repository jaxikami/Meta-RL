"""``python -m metarl <mode>``: see ``metarl.training.cli``."""

import sys

from metarl.training.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv))
