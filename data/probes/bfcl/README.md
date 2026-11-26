# BFCL subset (cached)

Source: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
Revision (pinned): `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda` (pulled 2026-11-26; the cached bytes were re-checked against this revision).
Licence: Apache-2.0 (dataset card `license: apache-2.0`).

| file | sha256 |
|---|---|
| `BFCL_v3_simple.json` | `fbc37b2ad252bf9af985582e0e07b456173fe627d957491472ea9cef5fb83158` |
| `BFCL_v3_irrelevance.json` | `975f51c51f688649fd190078efd87081241e0a326f9114a2ea3c1ca2440d8690` |
| `possible_answer_BFCL_v3_simple.json` | `2911a2bc00df82c4f999ffa64fedb0164cb88e96212ffa5972087eb91fd496ee` |

`possible_answer_BFCL_v3_simple.json` is `possible_answer/BFCL_v3_simple.json` from the repo.
The same data is in `SOURCE.json`, which the probe stamps on every measurement line.
Used by `promptchain/profiler/tool_probe.py`. `simple_363` is skipped at load time: its ground
truth names `find_closest` but its function is `restaurant_search.find_closest`.
