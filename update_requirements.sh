#!/bin/bash

# Remove old requirements.txt if it exists
rm -f requirements.txt

# Generate new requirements.txt from current environment
pip freeze > requirements.txt

echo "requirements.txt has been updated."