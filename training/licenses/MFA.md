# MFA boundary-trimming attribution

This integration uses **Montreal Forced Aligner 2.2.17**, by Montreal Corpus
Tools, under the **MIT license**. Copyright (c) 2016 Montreal Corpus Tools.
License: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/v2.2.17/LICENSE

The following unmodified Ukrainian MFA assets are by **Michael McAuliffe and
Morgan Sonderegger (2022)**, maintained by Montreal Forced Aligner, and published
under **Creative Commons Attribution 4.0 International (CC BY 4.0)**:

- Ukrainian MFA acoustic model **v2.0.0a**:
  https://mfa-models.readthedocs.io/en/latest/acoustic/Ukrainian/Ukrainian%20MFA%20acoustic%20model%20v2_0_0a.html
- Ukrainian MFA pronunciation dictionary **v2.0.0a**:
  https://mfa-models.readthedocs.io/en/latest/dictionary/Ukrainian/Ukrainian%20MFA%20dictionary%20v2_0_0a.html
- Ukrainian MFA G2P model **v2.0.0a**:
  https://mfa-models.readthedocs.io/en/latest/g2p/Ukrainian/Ukrainian%20MFA%20G2P%20model%20v2_0_0a.html

License: https://creativecommons.org/licenses/by/4.0/
Legal terms: https://creativecommons.org/licenses/by/4.0/legalcode

MIT and CC BY 4.0 permit commercial use. Preserve the software license notice
when redistributing MFA, and provide attribution, the CC BY license link, and
an indication of changes when sharing the covered assets or adaptations.
These notices do not imply endorsement by the original authors.

This experiment adds G2P-generated pronunciations for panel vocabulary to a
local copy of the dictionary. That expanded dictionary is a modified artifact;
the upstream acoustic/G2P weights are unchanged. Download URLs and SHA-256
digests are pinned in `training/conf/quality_mfa_assets.json`. Models and the
expanded dictionary remain in the ignored runtime directory. Dataset licenses
remain separate from these tool/model licenses.

MFA's runtime dependencies retain their own licenses (including Apache-2.0
Kaldi and GPL/LGPL SoX components); running those executables for alignment
does not make the audio an MFA software distribution. This file records the
selected upstream licensing, not a blanket license for the entire TTS stack.
