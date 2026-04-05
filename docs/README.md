# Documentation

This folder is the practical documentation for `tensor-network-editor`.

It is written with a user-first approach: start here if you want to use the
library, open the visual editor, save tensor-network designs, and generate
Python code without having to read the whole codebase first.

## Start here

- If you want a quick first success, read [getting-started.md](getting-started.md).
- If you want to understand the normal workflow in the editor, read
  [user-guide.md](user-guide.md).
- If you want to use the package directly from Python, read
  [python-api.md](python-api.md).
- If something is not working as expected, read
  [faq-troubleshooting.md](faq-troubleshooting.md).

## What this library gives you

`tensor-network-editor` is a local-first Python package for:

- building tensor-network diagrams visually in your browser
- saving them as JSON files
- loading them back later
- generating readable Python code for several backends

The browser interface is local to your machine. The package starts a local
server, opens a browser tab, and waits until you confirm or cancel the session.

## Typical workflow

1. Install the package in a virtual environment.
2. Launch the editor from the CLI or from Python.
3. Draw or edit your tensor network.
4. Confirm the session.
5. Save the abstract design as JSON and optionally generate Python code for a
   target engine.

## Which page should I read?

- Choose [getting-started.md](getting-started.md) if you are new to the project.
- Choose [user-guide.md](user-guide.md) if you already launched the editor and
  want to understand what the different features are for.
- Choose [python-api.md](python-api.md) if you prefer to integrate the package
  into scripts, notebooks, or larger workflows.

