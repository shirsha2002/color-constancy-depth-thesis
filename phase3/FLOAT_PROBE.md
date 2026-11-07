# Input-precision follow-up, before final depth-score inspection

7 November 2026. A pixel-only audit of the locked quantized input pipeline found
maximum relative luminance drift0.6284% (95th percentile0.2999%) between intended
chromatic and neutral controls. Float transforms themselves match luminance;
eight-bit rounding introduces the measured residual.

The original runs and all trained models remain preserved and unchanged. A
separate frozen-model probe therefore constructs the same two chromatic and
one exposure intervention in floating point **after** fixed spatial resampling
to the224-pixel encoder grid. All arms have identical geometry. The canonical
resampled source is clipped tovalidRGB before generating controls. No eight-bit
rounding follows the photometric interventions. The actual normalized encoder
tensor is inverted to check its input luminance, with a maximum relative
chromatic-control drift acceptance threshold5e-5.

This is a numerical intervention on real NYU images, not captured relighting.
The128scene set and reference depth are unchanged. Only the original frozen
depth model is tested; no checkpoint, architecture or hyperparameter is chosen
from these results. All five conditions and every scene are retained.

Both the locked eight-bit experiment and this additional precision diagnostic
are reported. Their absolute scores need not match because their resampling
and precision conventions differ. The purpose is to check the nuisance effect
under tighter brightness control, not to replace an unfavorable main result.
