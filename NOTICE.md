# Third-party components

This directory packages an existing Python hybrid docking implementation.

- Vina 1.2.7: bundled upstream source archive and isolated adapter; Apache-2.0 license in `native/LICENSE`.
- RTMScore: unchanged inference source and model1 checkpoint from the existing installation; license in `third_party/RTMScore/LICENSE`.
- The native library is a server-built Linux binary; its original build provenance is preserved. Rebuild it for incompatible systems.

The export removes historical experiment entry points and changes package imports/asset paths, not the search or scoring mathematics.

A minimal `RTMScore/__init__.py` was added to pin the bundled modules as a regular package. The worker verifies its package search path before loading model/feature code. The other bundled upstream modules and checkpoint retain their original bytes.
