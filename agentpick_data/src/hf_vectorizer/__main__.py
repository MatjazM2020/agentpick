"""Module entry points for CLI commands."""

import sys

# Vectorizer entry point
if __name__ == "__main__" and "vectorizer" in sys.argv[0]:
    from .vectorizer import main
    main()

# Query entry point
elif __name__ == "__main__" and "query" in sys.argv[0]:
    from .query import main
    main()
