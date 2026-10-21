# -*- coding: utf-8 -*-
"""Standard-profile registry: IEEE 802.3-2022 vs IEEE P802.3dj draft.

This module is the single authoritative separation between the *published*
IEEE 802.3-2022 TDECQ/TECQ measurement and the *draft* IEEE P802.3dj
reference-equalizer extension.  Every draft-only feature of the tool
(bounded multi-tap FFE candidate pool, one-tap DFE, tap-bound table,
differential limit) is gated by the profile selected here, and every output
JSON carries a ``standard`` marker naming the profile that produced it.

Feature provenance
------------------

IEEE 802.3-2022 (published standard)
    * TDECQ defined in Clause 121.8.5.3 (heritage: 802.3bs Clause 121 /
      802.3cd Clause 122; TECQ is the electrical-domain counterpart).
    * Reference receiver: a 5-tap, T/2-spaced FFE with unity gain (tap sum
      = 1); tap weights optimised (LMMSE) to minimise the noise at the
      measurement points for a target SER of 4.8e-4 per threshold.
    * Vertical histogram with a fixed OMA-anchored range
      (Pave +/- 2*OMA, 512 bins) per Clause 121.8.5.3.
    * Test patterns: SSPRQ (Clause 120.5.11 short stress pattern random
      quaternary, 65535 symbols) and the PRBS13Q-family dual-plane
      construction for the 128G MZM records.
    * Chromatic dispersion 0.8 ps/nm @ 1310 nm + receiver low-pass for the
      optical domain.
    * NO DFE and NO FFE tap-bound table anywhere in 802.3-2022.

IEEE P802.3dj draft (200G/lane; draft D2.1 task-force adoptions, see
``chayeb_3dj_02_2509`` "TDECQ Reference Equalizer Definition",
``ghiasi_3dj_01a_2509`` "Adjusting TDECQ Limit with Addition of 1T DFE")
    * Reference equalizer extended to a 15-tap FFE + one-tap DFE.
    * DFE feedback coefficient bound ``0 <= b(1) < 0.3`` (open upper bound;
      feedback referenced to OMA_outer/2).
    * FFE tap-limit table normalised to the main tap: main 0.8..2.5,
      |n|=1 -0.4..0.05, |n|=2 -0.1..0.2, |n|>=3 -0.1..0.1 (implemented in
      :mod:`tdecq_tool.ffe_bounds`); unit gain sum(w)=1; coupled
      differential limit ``|w(+1)/w(0) - b(1) - w(-1)/w(0)| <= 0.25``.
    * Structural caps: 3 pre-cursor / 13 post-cursor taps.
    * All of the above are DRAFT provisions - they are not part of
      IEEE 802.3-2022 and measurements made with this profile must not be
      reported as 802.3-2022 compliant.

The profiles are data, not code paths: the pipeline reads them to decide
which candidate machinery may run and which JSON fields may be emitted, so
the two modes cannot silently mix.
"""
from __future__ import annotations

STANDARD_2022 = "ieee8023-2022"
STANDARD_DJ_DRAFT = "ieee8023dj-draft"

DEFAULT_STANDARD = STANDARD_2022


class StandardProfile:
    """Immutable description of one standard profile."""

    __slots__ = ("name", "label", "draft", "allow_ffe_bounds", "allow_dfe",
                 "default_pre", "default_post", "tap_counts", "citation")

    def __init__(self, name, label, draft, allow_ffe_bounds, allow_dfe,
                 default_pre, default_post, tap_counts, citation):
        self.name = name
        self.label = label
        self.draft = bool(draft)
        self.allow_ffe_bounds = bool(allow_ffe_bounds)
        self.allow_dfe = bool(allow_dfe)
        self.default_pre = int(default_pre)
        self.default_post = int(default_post)
        self.tap_counts = tuple(int(x) for x in tap_counts)
        self.citation = str(citation)

    @property
    def default_tap_count(self):
        return self.default_pre + 1 + self.default_post

    def check_ffe_bounds_allowed(self):
        """Raise ValueError when the bounded FFE pool is not permitted."""
        if not self.allow_ffe_bounds:
            raise ValueError(
                "--ffe-bounds / the bounded multi-tap FFE candidate pool is "
                "an IEEE P802.3dj DRAFT feature (%s); it is not part of %s. "
                "Use --standard %s to enable it."
                % (STANDARD_DJ_DRAFT, self.name, STANDARD_DJ_DRAFT))

    def check_dfe_allowed(self):
        """Raise ValueError when the one-tap DFE is not permitted."""
        if not self.allow_dfe:
            raise ValueError(
                "--dfe-b1-unbounded / the one-tap DFE is an IEEE P802.3dj "
                "DRAFT feature (%s); it is not part of %s. Use --standard "
                "%s to enable it." % (STANDARD_DJ_DRAFT, self.name,
                                      STANDARD_DJ_DRAFT))

    def __repr__(self):
        return "StandardProfile(%r, draft=%r)" % (self.name, self.draft)


_PROFILES = {
    STANDARD_2022: StandardProfile(
        name=STANDARD_2022,
        label="IEEE 802.3-2022 TDECQ/TECQ (Clause 121.8.5.3 reference "
              "receiver: 5-tap FFE, no DFE)",
        draft=False,
        allow_ffe_bounds=False,
        allow_dfe=False,
        default_pre=0,
        default_post=4,
        tap_counts=(5,),
        citation="IEEE 802.3-2022 Clause 120.5.11 (SSPRQ) / 121.8.5.3 "
                 "(TDECQ); heritage IEEE 802.3bs king_3bs_01_0317",
    ),
    STANDARD_DJ_DRAFT: StandardProfile(
        name=STANDARD_DJ_DRAFT,
        label="IEEE P802.3dj DRAFT reference equalizer (15-tap FFE + 1T "
              "DFE, tap-limit table, b(1)<0.3) - NOT a published standard",
        draft=True,
        allow_ffe_bounds=True,
        allow_dfe=True,
        # 15-tap reference equalizer: 3 pre-cursors + main + 11 post.
        default_pre=3,
        default_post=11,
        tap_counts=(5, 7, 9, 15),
        citation="IEEE P802.3dj D2.1 adoptions: chayeb_3dj_02_2509, "
                 "ghiasi_3dj_01a_2509 (draft, subject to change)",
    ),
}


def get_profile(name=DEFAULT_STANDARD):
    """Return the :class:`StandardProfile` for ``name`` (exact key)."""
    try:
        return _PROFILES[str(name).strip().lower()]
    except KeyError:
        raise ValueError(
            "unknown standard profile %r; expected one of %s"
            % (name, ", ".join(sorted(_PROFILES)))) from None


def profile_names():
    """Sorted list of valid profile names."""
    return sorted(_PROFILES)
