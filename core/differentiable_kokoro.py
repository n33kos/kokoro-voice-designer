"""Differentiable reimplementation of Kokoro's forward pass.

Kokoro's `KModel.forward_with_tokens` is decorated with `@torch.no_grad()`, which
severs the autograd graph and makes gradient-based optimization of the style
vector impossible. Nothing inside it is inherently non-differentiable except the
duration rounding, so this module mirrors those ~30 lines with gradients enabled.

Kept deliberately line-for-line with upstream
(`.venv/lib/python3.12/site-packages/kokoro/model.py`) so future Kokoro releases
are easy to diff against. `test_differentiable.py` asserts bit-comparable output
against the stock path and is the regression check for that coupling.

Two things this exposes that the stock path discards:

- `duration` — the *continuous* per-token duration, before `round()` detaches it.
  This is the only differentiable handle on pacing.
- `F0_pred` / `N_pred` — the prosody predictor's pitch and energy contours, which
  give exact internal measurements instead of estimating them from the waveform.

Style vector layout (see `model.py:104` and `model.py:118`):

    ref_s[:, :128]  -> decoder      (timbre)
    ref_s[:, 128:]  -> predictor    (duration, pitch, energy — prosody)
"""

from dataclasses import dataclass

import torch


@dataclass
class TextContext:
    """Text-dependent tensors that don't vary with the style vector.

    `bert_dur`, `d_en` and `t_en` depend only on the token ids, so in an
    optimization loop over a fixed text they're constant and computed once.
    """

    input_ids: torch.LongTensor
    input_lengths: torch.LongTensor
    text_mask: torch.Tensor
    d_en: torch.Tensor
    t_en: torch.Tensor
    phonemes: str
    # True at token positions that carry actual speech sound. Stress marks,
    # spaces, punctuation and the BOS/EOS tokens legitimately get ~1 frame, so
    # any duration constraint has to exclude them.
    phoneme_mask: torch.Tensor
    # True at the tokens that become silence — spaces and punctuation. Stress
    # marks are excluded: they are not pauses.
    pause_mask: torch.Tensor
    # Punctuation split by strength. Kept separate because they behave
    # differently and averaging them hides the signal: one measured passage has
    # 82 spaces against 3 sentence ends, so a pooled "pause share" is dominated
    # by spaces and misses a 30% compression of the sentence breaks entirely.
    sentence_mask: torch.Tensor
    comma_mask: torch.Tensor


@dataclass
class DiffOutput:
    audio: torch.Tensor
    duration: torch.Tensor  # continuous, differentiable (pre-round)
    pred_dur: torch.LongTensor  # discrete frame counts (detached)
    f0_pred: torch.Tensor
    n_pred: torch.Tensor


class DifferentiableKokoro:
    """Wraps a `KModel` to expose a grad-enabled forward pass over `ref_s`."""

    def __init__(self, kmodel):
        self.model = kmodel
        self.device = kmodel.device
        # Optimizing the style vector, never the weights.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # -- text side -----------------------------------------------------------

    def encode_phonemes(self, phonemes: str) -> torch.LongTensor:
        """Phoneme string -> token ids. Mirrors `KModel.forward`."""
        vocab = self.model.vocab
        ids = [i for i in (vocab.get(p) for p in phonemes) if i is not None]
        return torch.LongTensor([[0, *ids, 0]]).to(self.device)

    @torch.no_grad()
    def build_context(self, phonemes: str) -> TextContext:
        """Precompute everything that doesn't depend on the style vector."""
        input_ids = self.encode_phonemes(phonemes)
        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=input_ids.device,
            dtype=torch.long,
        )
        text_mask = (
            torch.arange(input_lengths.max())
            .unsqueeze(0)
            .expand(input_lengths.shape[0], -1)
            .type_as(input_lengths)
        )
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(self.device)

        bert_dur = self.model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)
        t_en = self.model.text_encoder(input_ids, input_lengths, text_mask)

        # Token layout is [BOS, *phonemes, EOS], so index i+1 corresponds to
        # phonemes[i].
        non_speech = set(" ˈˌ,.;:!?…\"'()-")
        pauses = set(" ,.;:!?…")
        sentence_end = set(".!?…")
        commas = set(",;:")
        mask = torch.zeros(input_ids.shape[1], dtype=torch.bool, device=self.device)
        pause_mask = torch.zeros_like(mask)
        sentence_mask = torch.zeros_like(mask)
        comma_mask = torch.zeros_like(mask)
        for i, ch in enumerate(phonemes):
            if i + 1 >= mask.shape[0]:
                continue
            if ch not in non_speech:
                mask[i + 1] = True
            elif ch in pauses:
                pause_mask[i + 1] = True
                if ch in sentence_end:
                    sentence_mask[i + 1] = True
                elif ch in commas:
                    comma_mask[i + 1] = True

        return TextContext(
            input_ids=input_ids,
            input_lengths=input_lengths,
            text_mask=text_mask,
            d_en=d_en,
            t_en=t_en,
            phonemes=phonemes,
            phoneme_mask=mask,
            pause_mask=pause_mask,
            sentence_mask=sentence_mask,
            comma_mask=comma_mask,
        )

    # -- style side ----------------------------------------------------------

    def forward(
        self,
        ctx: TextContext,
        ref_s: torch.Tensor,
        speed: float = 1.0,
        decode: bool = True,
    ) -> DiffOutput:
        """Grad-enabled equivalent of `KModel.forward_with_tokens`.

        `ref_s` is [1, 256] and may require grad; everything returned in
        `DiffOutput` except `pred_dur` stays attached to it.

        `decode=False` stops before the vocoder and returns `audio=None`. Peak
        memory in the backward pass is dominated by retained vocoder activations
        at 24 kHz — roughly 0.75 GB per second of audio — so duration-only
        constraints can be applied at utterance lengths that would be far too
        expensive to render. `duration` is produced before the decoder and is
        unaffected.
        """
        predictor = self.model.predictor
        ref_s = ref_s.to(self.device)

        s = ref_s[:, 128:]
        d = predictor.text_encoder(ctx.d_en, s, ctx.input_lengths, ctx.text_mask)
        x, _ = predictor.lstm(d)
        duration = predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed

        # round()/long() sever the graph here — pacing gradients come from the
        # continuous `duration` above, not from this path.
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        # Alignment matrix is a constant once durations are fixed, so `en` and
        # `asr` stay differentiable w.r.t. d and t_en.
        indices = torch.repeat_interleave(
            torch.arange(ctx.input_ids.shape[1], device=self.device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (ctx.input_ids.shape[1], indices.shape[0]), device=self.device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        if not decode:
            return DiffOutput(audio=None, duration=duration, pred_dur=pred_dur,
                              f0_pred=None, n_pred=None)

        en = d.transpose(-1, -2) @ pred_aln_trg
        f0_pred, n_pred = predictor.F0Ntrain(en, s)
        asr = ctx.t_en @ pred_aln_trg
        audio = self.model.decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze()

        return DiffOutput(
            audio=audio,
            duration=duration,
            pred_dur=pred_dur,
            f0_pred=f0_pred,
            n_pred=n_pred,
        )

    # -- convenience ---------------------------------------------------------

    def phonemize(self, pipeline, text: str) -> str:
        """Run a KPipeline's g2p to get the phoneme string for `text`.

        Length matters: the row of a voice pack used for an utterance is
        `len(phonemes) - 1`.
        """
        _, tokens = pipeline.g2p(text)
        ps = "".join(
            (t.phonemes or "") + (" " if t.whitespace else "") for t in tokens
        ).strip()
        return ps
