import streamlit.web.cli as stcli
import os, sys


def resolve_path(path):
    resolved_path_t = os.path.join(os.path.realpath(os.path.dirname(sys.argv[0])), path)

    return resolved_path_t


if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("view/说明.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())
