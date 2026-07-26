# Expanded-v3 license review

## Decision rule

Use a source only when its dataset card or user grant permits commercial
training. Do not use a code license as proof for unrelated audio. A per-file
source needs license evidence for each selected file.

## Included sources

| Source | Decision | Evidence |
|---|---|---|
| Available Common Voice Ukrainian | INCLUDE | The local records use CC0. The user selected this available copy. |
| Google FLEURS Ukrainian | INCLUDE | The dataset card specifies CC BY 4.0. |
| TG Voices Ukrainian | INCLUDE | The dataset card specifies CC0. |
| Ukrainian dialect audio | INCLUDE | The dataset card specifies CC BY 4.0. |
| OpenTTS Lada | INCLUDE | The pinned dataset card specifies Apache 2.0. |
| OpenTTS Tetiana | INCLUDE | The pinned dataset card specifies Apache 2.0. |
| OpenTTS Mykyta | INCLUDE | The pinned dataset card specifies Apache 2.0. |
| UA-SER | INCLUDE | The dataset card specifies CC BY 4.0. |
| `robinhad/VOA-ukr` | INCLUDE | The user gave a commercial training grant. Raw redistribution is not permitted. |

## Skipped sources

| Source | Decision | Reason |
|---|---|---|
| `speech-uk/voice-of-america` | SKIP | The user requested this decision. |
| Shtooka mirror | SKIP | No complete per-file license manifest exists. |
| Lingua Libre | SKIP | No complete per-file license manifest exists. |
| Tatoeba | SKIP | No complete per-file license manifest exists. |
| LibriVox | SKIP | No per-record recording and text public-domain review exists. |
| Wikimedia Commons | SKIP | No complete per-file license manifest exists. |
| Direct Common Voice 26 | SKIP | The direct archive is not available. Use the selected local copy. |

The source registry contains the exact revisions and decisions.
