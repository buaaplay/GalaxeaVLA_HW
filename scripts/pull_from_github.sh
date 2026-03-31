#!/usr/bin/env bash
cd "$(dirname "$0")/.."
git fetch github
git merge github/main
