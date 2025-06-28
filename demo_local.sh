#!/bin/bash

printline() {
	echo "==========================================================="
}

printline

echo -n "Checking Python 3.10.12 or higher... "

VERSION=$(python3 -V | awk '{print $2}')
REQUIRED_VERSION="3.10.12"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "ERROR."
    echo "    (1) You need Python 3.10.12 or higher to run!"
    echo "    (2) Please, install Python 3.10.12 or higher, or use the Docker demo (demo_docker.sh)."
    printline
    exit 1
fi

echo "done."

printline
echo -n "Starting Virtual Environment ... "

python3 -m venv venv
source venv/bin/activate

echo "done."

printline
echo -n "Installing Python Requirements ... "

pip install -r src/requirements.txt

echo "done."
printline

echo ""

printline
echo "Running SigAPI ... "
echo ""

cd src
python3 main.py -d ../datasets/demo.csv --autocraft

echo ""
echo "done."
printline
