# Quickstart — five minutes, no API key

```bash
pip install -r requirements.txt
python tests/test_vigil.py          # expect: 24 passed, 0 failed
make demo                           # four runs on the bundled sample clip
make sweep                          # the threshold sweep
make figures                        # figures/ + results_table.csv
uvicorn server:app --port 8000      # open http://localhost:8000
```

In the dashboard: provider `mock`, URL `sample/demo_clip.avi`, press Start.
Frames kept, frames dropped and retention update live; descriptions fill the
timeline.

Then point it at something real:

```
https://www.youtube.com/watch?v=<a live stream id>
```

Switch the provider to `ollama` (free, local) or `gemini` (needs a key) once
the mock run looks right.

Next: read ROADMAP.md.
