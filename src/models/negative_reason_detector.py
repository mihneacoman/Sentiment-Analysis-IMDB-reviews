"""Rule-based negative review reason detection for IMDb-style reviews.

This module provides a lightweight baseline for identifying complaint reasons
in negative movie reviews without requiring a supervised reason-label dataset.
It is intended for weak labeling, manual inspection, clustering, and eventual
multi-label classification experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import html
import re
from typing import Any, Iterable


OTHER_LABEL = "other_uncertain"

NEGATIVE_CUES = (
    "awful",
    "bad",
    "boring",
    "cheap",
    "confusing",
    "disappointing",
    "dull",
    "failed",
    "flat",
    "hated",
    "horrible",
    "laughable",
    "mediocre",
    "mess",
    "poor",
    "pointless",
    "stupid",
    "terrible",
    "unconvincing",
    "underwhelming",
    "weak",
    "worse",
    "worst",
    "wasted",
)

NEGATION_TERMS = (
    "not",
    "never",
    "no",
    "hardly",
    "barely",
    "without",
    "neither",
    "nor",
)

CONTRAST_TERMS = ("but", "however", "although", "though", "yet")

INTENSIFIERS = (
    "absolutely",
    "completely",
    "deeply",
    "extremely",
    "incredibly",
    "painfully",
    "really",
    "so",
    "terribly",
    "totally",
    "truly",
    "very",
)


@dataclass(frozen=True)
class PatternSpec:
    """Weighted regex pattern used as evidence for a complaint reason."""

    regex: str
    weight: float
    evidence_type: str = "explicit"


def explicit(regex: str, weight: float = 1.0) -> PatternSpec:
    """Create a strong phrase-level evidence pattern."""

    return PatternSpec(regex=regex, weight=weight, evidence_type="explicit")


def weak(regex: str, weight: float = 0.45) -> PatternSpec:
    """Create a lower-confidence single-word or broad evidence pattern."""

    return PatternSpec(regex=regex, weight=weight, evidence_type="weak")


@dataclass(frozen=True)
class PatternMatch:
    """Concrete pattern match found in a sentence."""

    pattern: PatternSpec
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class TokenSpan:
    """Token text and character offsets within a sentence."""

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class ReasonSpec:
    """Configuration for one complaint reason."""

    label: str
    description: str
    patterns: tuple[PatternSpec, ...]


@dataclass
class ReasonEvidence:
    """Evidence collected for a single detected reason."""

    label: str
    score: float = 0.0
    evidence_count: int = 0
    explicit_evidence_count: int = 0
    weak_evidence_count: int = 0
    supporting_sentences: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)

    @property
    def heuristic_confidence(self) -> float:
        """Return a bounded, heuristic confidence estimate from rule evidence."""

        return min(1.0, round(self.score / 4.0, 3))

    def as_dict(self) -> dict[str, Any]:
        """Serialize evidence into a JSON-friendly dictionary."""

        heuristic_confidence = self.heuristic_confidence
        return {
            "label": self.label,
            "score": round(self.score, 3),
            "heuristic_confidence": heuristic_confidence,
            # Backward-compatible alias only; this is not a calibrated probability.
            "confidence": heuristic_confidence,
            "evidence_count": self.evidence_count,
            "explicit_evidence_count": self.explicit_evidence_count,
            "weak_evidence_count": self.weak_evidence_count,
            "supporting_sentences": self.supporting_sentences,
            "matched_patterns": self.matched_patterns,
        }


@dataclass(frozen=True)
class DetectionResult:
    """Structured output returned by NegativeReasonDetector."""

    labels: list[str]
    reasons: dict[str, dict[str, Any]]
    has_multiple_reasons: bool
    sentence_count: int
    normalized_text: str | None = None

    def to_dict(self, include_normalized_text: bool = False) -> dict[str, Any]:
        """Serialize the result into a JSON-, CSV-, or DataFrame-friendly dictionary."""

        result = {
            "labels": self.labels,
            "has_multiple_reasons": self.has_multiple_reasons,
            "sentence_count": self.sentence_count,
            "reason_scores": {
                label: reason["score"] for label, reason in self.reasons.items()
            },
            "heuristic_confidence": {
                label: reason["heuristic_confidence"]
                for label, reason in self.reasons.items()
            },
            "supporting_sentences": {
                label: reason["supporting_sentences"]
                for label, reason in self.reasons.items()
            },
            "matched_patterns": {
                label: reason["matched_patterns"] for label, reason in self.reasons.items()
            },
            "reasons": self.reasons,
        }
        if include_normalized_text and self.normalized_text is not None:
            result["normalized_text"] = self.normalized_text
        return result

    def as_dict(self, include_normalized_text: bool = False) -> dict[str, Any]:
        """Backward-compatible alias for to_dict()."""

        return self.to_dict(include_normalized_text=include_normalized_text)


DEFAULT_REASON_SPECS: tuple[ReasonSpec, ...] = (
    ReasonSpec(
        label="bad_acting",
        description="Performances are unconvincing, wooden, or distracting.",
        patterns=(
            explicit(r"\bbad acting\b", 1.2),
            explicit(r"\bpoor acting\b", 1.2),
            explicit(r"\blousy acting\b", 1.2),
            explicit(r"\bterrible acting\b", 1.4),
            explicit(r"\bawful acting\b", 1.4),
            explicit(r"\bhorrendous acting\b", 1.4),
            explicit(r"\bhammy acting\b", 1.1),
            explicit(r"\bflat acting\b", 1.1),
            explicit(r"\bwooden acting\b", 1.2),
            explicit(r"\bwooden performances?\b", 1.1),
            explicit(r"\bunconvincing performances?\b", 1.1),
            explicit(r"\bacting (?:was|is|were|are|felt|seemed) (?:even )?(?:bad|awful|terrible|horrendous|poor|weak|flat|wooden|unconvincing|worse)\b", 1.2),
            explicit(r"\bperformances? (?:were|are|is|was|felt|seemed) (?:bad|awful|terrible|weak|flat|wooden|unconvincing)\b", 1.2),
            explicit(r"\bactors? (?:were|are|was|is|felt|seemed) (?:bad|awful|terrible|weak|flat|wooden|unconvincing)\b", 1.1),
            explicit(r"\bterrible actors?\b", 1.2),
            explicit(r"\bseen better acting\b", 1.1),
            explicit(r"\bunbelievably wooden\b", 1.1),
            explicit(r"\bcast (?:was|is|felt|seemed) (?:bad|awful|terrible|weak|flat|wooden|unconvincing)\b", 1.0),
            explicit(r"\bcan not act\b", 1.2),
            explicit(r"\bcast (?:is|was|were|are) wasted\b", 1.1),
            explicit(r"\bwasted performances?\b", 1.1),
            explicit(r"\boveract(?:ed|ing)?\b", 0.9),
            explicit(r"\bdialogue and acting (?:were|are|was|is) (?:even )?worse\b", 1.1),
        ),
    ),
    ReasonSpec(
        label="weak_plot_bad_writing",
        description="The story, script, or writing is weak, illogical, or poorly built.",
        patterns=(
            explicit(r"\bweak plot\b", 1.2),
            explicit(r"\bbad plot\b", 1.2),
            explicit(r"\bthin plot\b", 1.0),
            explicit(r"\bbad screenplay\b", 1.2),
            explicit(r"\bbad script\b", 1.2),
            explicit(r"\bplot (?:was|is|felt|seemed) (?:weak|bad|awful|terrible|thin|messy|stupid|nonsensical|ridiculous)\b", 1.2),
            explicit(r"\bplot (?:was|is|felt|seemed) full of irrelevance\b", 1.2),
            explicit(r"\bplot (?:was|is|felt|seemed) inane\b", 1.2),
            explicit(r"\binconsistent plot\b", 1.1),
            explicit(r"\bplot is stupid\b", 1.2),
            explicit(r"\bplot is inane\b", 1.2),
            explicit(r"\bnot much (?:to|of) (?:a )?plot\b", 1.1),
            explicit(r"\bno plot\b", 1.1),
            explicit(r"\bno clear plot\b", 1.2),
            explicit(r"\black of (?:a good )?plot\b", 1.2),
            explicit(r"\black of (?:a )?story\b", 1.1),
            explicit(r"\black thereof\b", 1.1),
            explicit(r"\bpoorly written\b", 1.1),
            explicit(r"\bbadly written\b", 1.1),
            explicit(r"\bbad writing\b", 1.2),
            explicit(r"\bweak writing\b", 1.2),
            explicit(r"\bterrible writing\b", 1.4),
            explicit(r"\bscreenplay (?:was|is|felt|seemed) (?:bad|awful|terrible|weak|messy|lazy)\b", 1.2),
            explicit(r"\bscript (?:was|is|felt|seemed) (?:weak|bad|awful|terrible|messy|lazy)\b", 1.1),
            explicit(r"\blazy writing\b", 1.0),
            explicit(r"\bplot holes?\b", 0.9),
            explicit(r"\bdoes not make sense\b", 1.1),
            explicit(r"\bdoesn't make sense\b", 1.1),
            explicit(r"\bmakes? no sense\b", 1.0),
            explicit(r"\bplot makes? no sense\b", 1.2),
            explicit(r"\bstory makes? no sense\b", 1.2),
            explicit(r"\bstoryline (?:was|is|felt|seemed) (?:almost |nearly |basically )?incomprehensible\b", 1.2),
            explicit(r"\bleads? to nowhere\b", 1.1),
            explicit(r"\bludicrous sub-?plot\b", 1.1),
            explicit(r"\bbeyond-?heavy-?handed\b", 1.2),
            explicit(r"\bheavy-?handed\b", 1.0),
            explicit(r"\bcliche(?:d|s)?\b", 0.9),
            explicit(r"\bhackneyed\b", 1.0),
            explicit(r"\buninspired\b", 1.0),
            explicit(r"\black(?:s|ed)? imagination\b", 1.1),
            explicit(r"\bcontrived\b", 1.0),
            explicit(r"\bwriters? lose track\b", 1.2),
            explicit(r"\bscript loses track\b", 1.2),
            explicit(r"\bno idea how to resolve\b", 1.2),
            explicit(r"\bno one could tell me what the actual storyline was\b", 1.3),
            explicit(r"\bwhat the actual storyline was\b", 1.1),
            explicit(r"\bdevelopment of the story\b", 1.0),
            explicit(r"\bassist the development of the story\b", 1.1),
            explicit(r"\bwriter feels at a loss\b", 1.2),
            explicit(r"\bstoryline was unclear\b", 1.2),
        ),
    ),
    ReasonSpec(
        label="poor_direction_execution",
        description="Direction, production choices, or execution fail to realize the material.",
        patterns=(
            explicit(r"\bpoorly directed\b", 1.2),
            explicit(r"\bbadly directed\b", 1.2),
            explicit(r"\bboringly directed\b", 1.1),
            explicit(r"\bpoorly executed\b", 1.2),
            explicit(r"\bbadly executed\b", 1.2),
            explicit(r"\binept adaptation\b", 1.2),
            explicit(r"\bdirector fails\b", 1.2),
            explicit(r"\bfails on all possible accounts\b", 1.2),
            explicit(r"\bchoices made in the production are extremely poor\b", 1.3),
            explicit(r"\bproduction choices (?:are|were|seem|seemed) (?:extremely |very |so )?poor\b", 1.3),
            explicit(r"\bnot fully realized\b", 1.1),
            explicit(r"\bwriter feels at a loss\b", 1.0),
            explicit(r"\bat a loss to where to go\b", 1.0),
            explicit(r"\bno idea how to resolve\b", 1.0),
            explicit(r"\bquality and substance over just style\b", 1.1),
            explicit(r"\bstyle over substance\b", 1.2),
            explicit(r"\btrying too hard\b", 1.0),
        ),
    ),
    ReasonSpec(
        label="tonal_mismatch",
        description="The film's genre, tone, or mixture of elements does not work.",
        patterns=(
            explicit(r"\btoo over-?sexed to be (?:a |an )?(?:straight )?horror\b", 1.3),
            explicit(r"\btoo gruesome to work as (?:a |an )?sex flick\b", 1.3),
            explicit(r"\btoo gruesome to work\b", 1.1),
            explicit(r"\btoo serious to be funny\b", 1.2),
            explicit(r"\btoo silly to be serious\b", 1.2),
            explicit(r"\bsupposed to be lighthearted\b", 1.1),
            explicit(r"\bcomes off as creepy\b", 1.2),
            explicit(r"\bdownright creepy\b", 1.1),
            explicit(r"\bnot scary\b", 1.1),
            explicit(r"\bnot an effective comedy\b", 1.2),
            explicit(r"\bneither clever nor romantic\b", 1.2),
            explicit(r"\bdoes not work as (?:a |an )?comedy\b", 1.2),
            explicit(r"\bdoes not work as (?:a |an )?horror\b", 1.2),
            explicit(r"\bdoes not work as (?:a |an )?drama\b", 1.2),
            explicit(r"\bdoes not work as (?:a |an )?thriller\b", 1.2),
            explicit(r"\btonal mismatch\b", 1.2),
            explicit(r"\btone is all over the place\b", 1.2),
            explicit(r"\bdoes not know what it wants to be\b", 1.2),
            explicit(r"\bstruggles to decide whether\b", 1.1),
            explicit(r"\bno balance\b", 1.0),
            explicit(r"\bthere needs to be a balance\b", 1.1),
            explicit(r"\bthe balance is off\b", 1.1),
        ),
    ),
    ReasonSpec(
        label="factual_inaccuracy",
        description="The film is criticized for historical or factual inaccuracy.",
        patterns=(
            explicit(r"\bfactually incorrect\b", 1.3),
            explicit(r"\bhistorically inaccurate\b", 1.3),
            explicit(r"\bhistorical inaccuracies\b", 1.2),
            explicit(r"\bdemonstrably incorrect\b", 1.3),
            explicit(r"\bthey did not happen\b", 1.1),
            explicit(r"\bdid not happen\b", 1.0),
            explicit(r"\breliance on facts rapidly decays\b", 1.3),
            explicit(r"\btoo many gross errors\b", 1.3),
            explicit(r"\bgross errors\b", 1.1),
            explicit(r"\bshameless invention(?:s)?\b", 1.2),
            explicit(r"\bbased on actual events\b", 1.0),
            explicit(r"\bcritical facts\b", 1.1),
            explicit(r"\bfantasy of this film\b", 1.1),
            explicit(r"\bpretends to reenact\b", 1.2),
            explicit(r"\bfactual errors\b", 1.2),
            explicit(r"\binaccurate portrayal\b", 1.2),
            explicit(r"\binaccurate depiction\b", 1.2),
        ),
    ),
    ReasonSpec(
        label="boring_slow_pacing",
        description="The film feels boring, slow, repetitive, or too long.",
        patterns=(
            weak(r"\bboring\b", 0.55),
            weak(r"\bdull\b", 0.5),
            weak(r"\btedious\b", 0.55),
            weak(r"\bslow\b", 0.35),
            explicit(r"\bextremely boring\b", 1.2),
            explicit(r"\bincredibly boring\b", 1.2),
            explicit(r"\bstill boring\b", 1.0),
            explicit(r"\bboringly directed\b", 1.0),
            explicit(r"\btiresome\b", 1.0),
            explicit(r"\bsnoozer\b", 1.0),
            explicit(r"\btoo long\b", 0.9),
            explicit(r"\bway too long\b", 1.1),
            explicit(r"\bdragged\b", 1.0),
            explicit(r"\bdrags\b", 0.9),
            explicit(r"\bslow paced\b", 1.1),
            explicit(r"\bslow-paced\b", 1.1),
            explicit(r"\bpacing (?:was|is|felt|seemed) (?:bad|awful|terrible|slow|uneven|poor)\b", 1.2),
            explicit(r"\bnothing happens\b", 1.1),
            explicit(r"\bfell asleep\b", 1.1),
            explicit(r"\bfall asleep\b", 1.1),
            weak(r"\brepetitive\b", 0.55),
            explicit(r"\bkept waiting\b", 0.8),
            explicit(r"\bwaste of time\b", 1.2),
            explicit(r"\bwaste of my time\b", 1.2),
            explicit(r"\bwaste time\b", 1.1),
            explicit(r"\bdo not waste your time\b", 1.3),
            explicit(r"\bdon't waste your time\b", 1.3),
            explicit(r"\bfast forward(?:ed|ing)?\b", 1.1),
            explicit(r"\bwaiting for it to end\b", 1.1),
            explicit(r"\bbored to bits\b", 1.1),
            explicit(r"\bboring nonsense\b", 1.1),
            explicit(r"\bnot watchable\b", 1.1),
            explicit(r"\bdevoid of any sort of inspiration\b", 1.1),
            explicit(r"\bseems? like (?:an )?eternity\b", 1.0),
            explicit(r"\bfelt like (?:an )?eternity\b", 1.0),
            explicit(r"\bdifficult to watch\b", 1.1),
            explicit(r"\bhard to sit through\b", 1.1),
            explicit(r"\bcould not sit through\b", 1.1),
            explicit(r"\bcouldn't sit through\b", 1.1),
            explicit(r"\bsuffering through\b", 1.0),
        ),
    ),
    ReasonSpec(
        label="disappointing_ending",
        description="The ending, finale, or climax is unsatisfying.",
        patterns=(
            explicit(r"\bbad ending\b", 1.2),
            explicit(r"\bweak ending\b", 1.2),
            explicit(r"\bterrible ending\b", 1.4),
            explicit(r"\bawful ending\b", 1.4),
            explicit(r"\bdisappointing ending\b", 1.3),
            explicit(r"\bending (?:was|is|felt|seemed) (?:bad|awful|terrible|weak|rushed|cheap|disappointing|unsatisfying)\b", 1.3),
            explicit(r"\bfinale (?:was|is|felt|seemed) (?:bad|awful|terrible|weak|rushed|cheap|disappointing|unsatisfying)\b", 1.2),
            explicit(r"\bclimax (?:was|is|felt|seemed) (?:bad|awful|terrible|weak|rushed|cheap|disappointing|underwhelming)\b", 1.1),
            explicit(r"\brushed ending\b", 1.2),
            explicit(r"\bunsatisfying ending\b", 1.2),
        ),
    ),
    ReasonSpec(
        label="failed_expectations",
        description="The movie fails to meet expectations set by premise, hype, cast, or creators.",
        patterns=(
            explicit(r"\bexpected (?:so much more|much more|more|better|too much)\b", 1.1),
            explicit(r"\bnot what i expected\b", 1.2),
            explicit(r"\bdid not live up to\b", 1.3),
            explicit(r"\bdidn't live up to\b", 1.3),
            explicit(r"\bfails? to live up to\b", 1.3),
            explicit(r"\blived up to none\b", 1.3),
            explicit(r"\bhuge disappointment\b", 1.3),
            explicit(r"\bmajor disappointment\b", 1.3),
            explicit(r"\bsorely disappointed\b", 1.2),
            explicit(r"\bexpecting a lot better\b", 1.2),
            explicit(r"\bexpected a lot better\b", 1.2),
            explicit(r"\bwaste of potential\b", 1.2),
            explicit(r"\bwasted potential\b", 1.2),
            explicit(r"\bwasted opportunit(?:y|ies)\b", 1.2),
            explicit(r"\bwasted talent\b", 1.1),
            explicit(r"\bwasted cast\b", 1.1),
            explicit(r"\boverhyped\b", 0.9),
            weak(r"\bunderwhelming\b", 0.65),
            explicit(r"\b(?:really )?looking forward to\b", 1.0),
            explicit(r"\breally wanted to like\b", 1.1),
            explicit(r"\bwanted to like this\b", 1.1),
            explicit(r"\bwanted this movie to be good\b", 1.2),
            explicit(r"\bwanted this film to be good\b", 1.2),
            explicit(r"\bhad high hopes\b", 1.1),
            explicit(r"\bexpectations were too high\b", 1.2),
            explicit(r"\bexpectations were forced too high\b", 1.2),
            explicit(r"\bseemed promising\b", 1.0),
            explicit(r"\bpromising premise\b", 1.0),
            explicit(r"\bleft me cold\b", 1.1),
            explicit(r"\bleft me dry\b", 1.1),
            explicit(r"\bleft a little dry\b", 1.1),
            explicit(r"\bbegan to lose faith\b", 1.1),
            explicit(r"\blet me down\b", 1.2),
            explicit(r"\blet down\b", 1.0),
            explicit(r"\bletdown\b", 1.0),
            explicit(r"\bbubble (?:got )?burst\b", 1.1),
            explicit(r"\bcould have been\b", 0.9),
            explicit(r"\bcould have been better\b", 1.1),
            explicit(r"\bshould have been\b", 0.9),
            explicit(r"\bshould have been better\b", 1.1),
            explicit(r"\bdoes not do justice\b", 1.1),
            explicit(r"\bdoesn't do justice\b", 1.1),
            weak(r"\bdisappointed\b", 0.65),
        ),
    ),
    ReasonSpec(
        label="weak_characters",
        description="Characters are flat, unlikeable, underdeveloped, or poorly motivated.",
        patterns=(
            explicit(r"\bweak characters?\b", 1.2),
            explicit(r"\bbad characters?\b", 1.2),
            explicit(r"\bflat characters?\b", 1.1),
            explicit(r"\bone dimensional characters?\b", 1.1),
            explicit(r"\bone-dimensional characters?\b", 1.1),
            explicit(r"\bunderdeveloped characters?\b", 1.1),
            explicit(r"\bpoorly developed characters?\b", 1.1),
            explicit(r"\bthin characters?\b", 1.0),
            explicit(r"\bunlikable characters?\b", 1.0),
            explicit(r"\bunlikeable characters?\b", 1.0),
            explicit(r"\bannoying characters?\b", 1.0),
            explicit(r"\bobnoxious\b", 1.0),
            explicit(r"\bless than relatable\b", 1.0),
            explicit(r"\bunrelatable\b", 1.0),
            explicit(r"\bcharacters? (?:were|are|was|is|felt|seemed) (?:weak|bad|flat|boring|annoying|obnoxious|unlikable|unlikeable|unrelatable|underdeveloped|thin|one-dimensional)\b", 1.2),
            explicit(r"\bno character development\b", 1.2),
            explicit(r"\bcharacter development (?:was|is|felt|seemed) (?:weak|bad|poor|missing)\b", 1.2),
            explicit(r"\bdo not care about the characters\b", 1.2),
            explicit(r"\bcould not care about the characters\b", 1.2),
            explicit(r"\bdid not care about the characters\b", 1.2),
            explicit(r"\bdidn't care about the characters\b", 1.2),
            explicit(r"\bcouldn't care about the characters\b", 1.2),
            explicit(r"\bno connection to the characters\b", 1.2),
            explicit(r"\bcharacters were annoying\b", 1.1),
            explicit(r"\bno chemistry\b", 1.1),
            explicit(r"\bprefer brighter company\b", 1.0),
            explicit(r"\bno one to root for\b", 1.0),
        ),
    ),
    ReasonSpec(
        label="bad_dialogue",
        description="Dialogue is awkward, cheesy, unnatural, or poorly written.",
        patterns=(
            explicit(r"\bbad dialogue\b", 1.2),
            explicit(r"\bawful dialogue\b", 1.4),
            explicit(r"\bterrible dialogue\b", 1.4),
            explicit(r"\bweak dialogue\b", 1.2),
            explicit(r"\batrocious dialogue\b", 1.4),
            explicit(r"\bflat dialogue\b", 1.1),
            explicit(r"\bbad screenplay and dialogue\b", 1.3),
            explicit(r"\blame dialogue\b", 1.1),
            explicit(r"\bno dialogue\b", 1.1),
            explicit(r"\bno dialog\b", 1.1),
            explicit(r"\bcheesy dialogue\b", 1.0),
            explicit(r"\bcringey dialogue\b", 1.0),
            explicit(r"\bcringe dialogue\b", 1.0),
            explicit(r"\bdialogue (?:was|is|were|are|felt|seemed) (?:even |painfully |really |very |so |extremely |incredibly |terribly )?(?:bad|awful|atrocious|terrible|weak|flat|worse|cheesy|cringey|stilted|unnatural)\b", 1.2),
            explicit(r"\bdialog (?:was|is|were|are|felt|seemed) (?:even |painfully |really |very |so |extremely |incredibly |terribly )?(?:bad|awful|atrocious|terrible|weak|flat|worse|cheesy|cringey|stilted|unnatural)\b", 1.2),
            explicit(r"\bdialogue and acting (?:were|are|was|is) (?:even )?worse\b", 1.1),
            explicit(r"\bstilted dialogue\b", 1.0),
            explicit(r"\bunnatural dialogue\b", 1.0),
            explicit(r"\blines (?:were|are|felt|seemed) (?:bad|awful|terrible|cheesy|cringey|stilted)\b", 1.0),
            explicit(r"\bterrible lines\b", 1.1),
            explicit(r"\bbad lines\b", 1.1),
            explicit(r"\ball the lines sound\b", 1.1),
            explicit(r"\bstupid lines\b", 1.1),
        ),
    ),
    ReasonSpec(
        label="confusing_story",
        description="The story is hard to follow, incoherent, or confusing.",
        patterns=(
            weak(r"\bconfusing\b", 0.55),
            weak(r"\bconfused\b", 0.45),
            weak(r"\bincoherent\b", 0.6),
            explicit(r"\bhard to follow\b", 1.1),
            explicit(r"\bdifficult to follow\b", 1.1),
            explicit(r"\bcould not follow\b", 1.2),
            explicit(r"\bcouldn't follow\b", 1.2),
            explicit(r"\bmade no sense\b", 1.1),
            explicit(r"\bmakes no sense\b", 1.1),
            explicit(r"\bno sense\b", 0.9),
            explicit(r"\bplot (?:was|is|felt|seemed) (?:confusing|incoherent|messy|hard to follow)\b", 1.2),
            explicit(r"\bstory (?:was|is|felt|seemed) (?:confusing|incoherent|messy|hard to follow)\b", 1.2),
        ),
    ),
    ReasonSpec(
        label="poor_visuals_effects",
        description="Visuals, CGI, sets, makeup, or effects look poor or cheap.",
        patterns=(
            explicit(r"\bbad visuals?\b", 1.2),
            explicit(r"\bpoor visuals?\b", 1.2),
            explicit(r"\bcheap visuals?\b", 1.1),
            explicit(r"\bbad effects?\b", 1.2),
            explicit(r"\bpoor effects?\b", 1.2),
            explicit(r"\bcheap effects?\b", 1.1),
            explicit(r"\bbad special effects?\b", 1.3),
            explicit(r"\bpoor special effects?\b", 1.3),
            explicit(r"\bterrible special effects?\b", 1.4),
            explicit(r"\bcheap special effects?\b", 1.2),
            explicit(r"\bcheesy effects?\b", 1.1),
            explicit(r"\bspecial effects? (?:were|are|was|is|felt|seemed|looked|looks) (?:bad|awful|terrible|cheap|fake|poor|mundane|lame)\b", 1.2),
            explicit(r"\bspecial effects? (?:were|are|was|is|felt|seemed|looked|looks) not impressive\b", 1.2),
            explicit(r"\bso fake\b", 1.0),
            explicit(r"\bgrade-z effects?\b", 1.2),
            explicit(r"\bbad cgi\b", 1.2),
            explicit(r"\bpoor cgi\b", 1.2),
            explicit(r"\bterrible cgi\b", 1.4),
            explicit(r"\bawful cgi\b", 1.4),
            explicit(r"\bterrible digital animation\b", 1.4),
            explicit(r"\bugly animation\b", 1.2),
            explicit(r"\bgraphics look fake\b", 1.2),
            explicit(r"\beffects? look fake\b", 1.2),
            explicit(r"\beffects? looked fake\b", 1.2),
            explicit(r"\bcgi looks fake\b", 1.2),
            explicit(r"\bcgi looked fake\b", 1.2),
            explicit(r"\bcgi (?:was|is|felt|seemed|looked|looks) (?:bad|awful|terrible|cheap|fake|poor)\b", 1.2),
            explicit(r"\bcgi[- ]?effects? (?:was|is|were|are|felt|seemed|look|looked|looks) (?:bad|awful|terrible|cheap|fake|poor)\b", 1.2),
            explicit(r"\beffects? (?:were|are|was|is|felt|seemed|looked|looks) (?:bad|awful|terrible|cheap|fake|poor)\b", 1.2),
            explicit(r"\blooked cheap\b", 0.8),
            explicit(r"\bfake looking\b", 0.8),
        ),
    ),
    ReasonSpec(
        label="not_funny",
        description="A comedy or comic elements fail to be funny or humorous.",
        patterns=(
            explicit(r"\bnot funny\b", 1.2),
            explicit(r"\bnot funny at all\b", 1.4),
            explicit(r"\bunfunny\b", 1.2),
            explicit(r"\bnot amusing\b", 1.2),
            explicit(r"\bnot even (?:a bit )?amusing\b", 1.2),
            explicit(r"\bis not funny\b", 1.2),
            explicit(r"\bwas not funny\b", 1.2),
            explicit(r"\bnothing was funny\b", 1.3),
            explicit(r"\bnothing was humorous\b", 1.3),
            explicit(r"\bwhere was the joke\b", 1.2),
            explicit(r"\bjokes? (?:are|were|was|is) old\b", 1.1),
            explicit(r"\bjokes? (?:are|were|was|is) lame\b", 1.1),
            explicit(r"\blame jokes?\b", 1.1),
            explicit(r"\bdid not laugh\b", 1.2),
            explicit(r"\bdidn't laugh\b", 1.2),
            explicit(r"\bnot an effective comedy\b", 1.3),
            explicit(r"\bfails? as (?:a |an )?comedy\b", 1.3),
        ),
    ),
    ReasonSpec(
        label="poor_production_quality",
        description="The filmmaking craft, production values, or technical execution look amateurish.",
        patterns=(
            explicit(r"\bbadly made movie\b", 1.3),
            explicit(r"\bbadly made film\b", 1.3),
            explicit(r"\bpoorly made\b", 1.2),
            explicit(r"\bextremely amateur\b", 1.2),
            explicit(r"\blow budget\b", 1.0),
            explicit(r"\blow-budget\b", 1.0),
            explicit(r"\bpoor production values?\b", 1.3),
            explicit(r"\bawful production values?\b", 1.3),
            explicit(r"\bbad production values?\b", 1.3),
            explicit(r"\bterrible production values?\b", 1.3),
            explicit(r"\bcheap production values?\b", 1.2),
            explicit(r"\blow production values?\b", 1.2),
            explicit(r"\bcamera work\b", 1.0),
            explicit(r"\bbad camera work\b", 1.3),
            explicit(r"\bhome video camera\b", 1.2),
            explicit(r"\bshot on (?:a )?home video camera\b", 1.3),
            explicit(r"\bcamera angles? (?:that )?made me feel sick\b", 1.2),
            explicit(r"\bno cinematography\b", 1.2),
            explicit(r"\bediting blunders?\b", 1.2),
            explicit(r"\bbad editing\b", 1.2),
            explicit(r"\bpoor editing\b", 1.2),
            explicit(r"\bbad sound\b", 1.2),
            explicit(r"\bpoor sound\b", 1.2),
            explicit(r"\bshaky sound\b", 1.1),
            explicit(r"\bsound is awful\b", 1.2),
            explicit(r"\bsound was awful\b", 1.2),
            explicit(r"\bpoor sets?\b", 1.1),
            explicit(r"\bgrade school props\b", 1.2),
            explicit(r"\bcraft is too lacking\b", 1.2),
            explicit(r"\bpoor costuming\b", 1.1),
            explicit(r"\bbadly framed\b", 1.1),
            explicit(r"\bshot on video\b", 1.1),
        ),
    ),
    ReasonSpec(
        label="generic_unoriginal",
        description="The movie feels generic, predictable, derivative, or unoriginal.",
        patterns=(
            weak(r"\bgeneric\b", 0.55),
            weak(r"\bunoriginal\b", 0.6),
            weak(r"\bderivative\b", 0.55),
            weak(r"\bpredictable\b", 0.45),
            weak(r"\bcliche(?:d|s)?\b", 0.5),
            weak(r"\bformulaic\b", 0.55),
            explicit(r"\bnothing new\b", 1.0),
            explicit(r"\bseen it all before\b", 1.1),
            explicit(r"\bwhere has this setup been used before\b", 1.1),
            explicit(r"\bwhere has(?:n't| not) this setup been used before\b", 1.1),
            explicit(r"\bstandard fare\b", 1.0),
            explicit(r"\bcopy of\b", 0.8),
            explicit(r"\brip[ -]?off\b", 1.0),
            explicit(r"\bby the numbers\b", 1.0),
            explicit(r"\bpaint by numbers\b", 1.0),
            explicit(r"\bpredictable plot\b", 1.0),
        ),
    ),
)


class NegativeReasonDetector:
    """Detect likely complaint reasons in negative movie reviews.

    The detector is intentionally transparent: every label is assigned from
    sentence-level evidence matched by category-specific regular expressions
    and negative sentiment cues. heuristic_confidence values are rule-based
    scores, not calibrated probabilities. The legacy confidence field is kept
    only as a backward-compatible alias. This is a strong baseline for weak
    supervision, but it should not be treated as a replacement for manual
    validation.
    """

    def __init__(
        self,
        reason_specs: Iterable[ReasonSpec] = DEFAULT_REASON_SPECS,
        min_score: float = 1.0,
        weak_min_score: float = 1.25,
        max_supporting_sentences: int = 3,
    ) -> None:
        self.reason_specs = tuple(reason_specs)
        self.min_score = min_score
        self.weak_min_score = weak_min_score
        self.max_supporting_sentences = max_supporting_sentences
        self._compiled_patterns = {
            spec.label: tuple(
                (pattern, re.compile(pattern.regex, re.IGNORECASE))
                for pattern in spec.patterns
            )
            for spec in self.reason_specs
        }

    def analyze_review(
        self,
        review_text: object,
        include_normalized_text: bool = False,
    ) -> DetectionResult:
        """Analyze a single review and return detected complaint reasons.

        Args:
            review_text: Raw review text. Non-string values are handled safely.
            include_normalized_text: Include normalized text in the returned
                result for debugging/export auditing. Defaults to False to keep
                exported results compact.

        Returns:
            A DetectionResult containing labels, scores, supporting snippets,
            and metadata useful for downstream analysis.
        """

        normalized_text = self.normalize_text(review_text)
        sentences = self.split_sentences(normalized_text)

        if not normalized_text or not sentences:
            return self._uncertain_result(
                normalized_text=normalized_text,
                sentence_count=0,
                include_normalized_text=include_normalized_text,
            )

        evidence_by_label = {
            spec.label: ReasonEvidence(label=spec.label) for spec in self.reason_specs
        }

        for sentence in sentences:
            sentence_negative_score = self._negative_cue_score(sentence)

            for spec in self.reason_specs:
                matches = self._match_patterns(spec.label, sentence)
                if not matches:
                    continue

                valid_matches = [
                    match for match in matches if not self._is_negated_match(sentence, match)
                ]
                if not valid_matches:
                    continue

                score = self._score_matches(
                    matches=valid_matches,
                    sentence_negative_score=sentence_negative_score,
                )
                self._add_evidence(
                    evidence=evidence_by_label[spec.label],
                    sentence=sentence,
                    matches=valid_matches,
                    score=score,
                )

        detected = {
            label: evidence.as_dict()
            for label, evidence in evidence_by_label.items()
            if self._passes_detection_threshold(evidence)
        }

        if not detected:
            return self._uncertain_result(
                normalized_text=normalized_text,
                sentence_count=len(sentences),
                include_normalized_text=include_normalized_text,
            )

        labels = sorted(
            detected,
            key=lambda label: (
                detected[label]["score"],
                detected[label]["evidence_count"],
                label,
            ),
            reverse=True,
        )

        ordered_reasons = {label: detected[label] for label in labels}
        return DetectionResult(
            labels=labels,
            reasons=ordered_reasons,
            has_multiple_reasons=len(labels) > 1,
            sentence_count=len(sentences),
            normalized_text=normalized_text if include_normalized_text else None,
        )

    def analyze_reviews(
        self,
        reviews: Iterable[str],
        include_normalized_text: bool = False,
    ) -> list[DetectionResult]:
        """Analyze a batch of reviews using the same rules as analyze_review()."""

        return [
            self.analyze_review(
                review,
                include_normalized_text=include_normalized_text,
            )
            for review in reviews
        ]

    @staticmethod
    def normalize_text(review_text: object) -> str:
        """Normalize raw review text while preserving sentence boundaries."""

        if not isinstance(review_text, str):
            return ""

        text = html.unescape(review_text)
        text = re.sub(r"<br\s*/?>", ". ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = re.sub(r"\b(can't|cannot)\b", "can not", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(won't)\b", "will not", text, flags=re.IGNORECASE)
        text = re.sub(r"\b([a-z]+)n't\b", r"\1 not", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Split text into simple sentence-like units."""

        if not text:
            return []

        sentence_candidates = re.split(
            r"(?<=[.!?])\s+|;+\s*|(?:\s+-\s+)|\s+(?:but|however)\s+",
            text,
            flags=re.IGNORECASE,
        )
        return [sentence.strip(" \t\n\r\"'") for sentence in sentence_candidates if sentence.strip()]

    def _match_patterns(self, label: str, sentence: str) -> list[PatternMatch]:
        matches: list[PatternMatch] = []
        for pattern_spec, compiled_pattern in self._compiled_patterns[label]:
            for match in compiled_pattern.finditer(sentence):
                matches.append(
                    PatternMatch(
                        pattern=pattern_spec,
                        start=match.start(),
                        end=match.end(),
                        text=match.group(0),
                    )
                )
        return matches

    def _negative_cue_score(self, sentence: str) -> float:
        tokens = self._tokenize(sentence)
        if not tokens:
            return 0.0

        score = 0.0
        for index, token in enumerate(tokens):
            if token not in NEGATIVE_CUES:
                continue

            previous_tokens = tokens[max(0, index - 3) : index]
            if any(term in previous_tokens for term in NEGATION_TERMS):
                score -= 0.5
            else:
                score += 0.5

            if any(term in previous_tokens for term in INTENSIFIERS):
                score += 0.25

        return max(0.0, score)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z']+", text.lower())

    @staticmethod
    def _token_spans(text: str) -> list[TokenSpan]:
        return [
            TokenSpan(match.group(0).lower(), match.start(), match.end())
            for match in re.finditer(r"[a-zA-Z']+", text)
        ]

    def _is_negated_match(self, sentence: str, match: PatternMatch) -> bool:
        tokens = self._token_spans(sentence)
        if not tokens:
            return False

        match_token_indexes = [
            index for index, token in enumerate(tokens) if token.start < match.end and token.end > match.start
        ]
        if not match_token_indexes:
            return False

        first_match_token = match_token_indexes[0]
        previous_window = tokens[max(0, first_match_token - 4) : first_match_token]
        previous_terms = {token.text for token in previous_window}

        if not previous_terms.intersection(NEGATION_TERMS):
            return False

        negation_tokens = [token for token in previous_window if token.text in NEGATION_TERMS]
        nearest_negation = negation_tokens[-1]
        text_between = sentence[nearest_negation.end : match.start]
        if re.search(r"[,.;:!?]", text_between):
            return False

        contrast_between_negation_and_match = any(
            token.text in CONTRAST_TERMS for token in previous_window
        )
        return not contrast_between_negation_and_match

    @staticmethod
    def _score_matches(matches: list[PatternMatch], sentence_negative_score: float) -> float:
        explicit_score = sum(
            match.pattern.weight for match in matches if match.pattern.evidence_type == "explicit"
        )
        weak_score = sum(
            match.pattern.weight for match in matches if match.pattern.evidence_type == "weak"
        )

        if explicit_score > 0:
            context_bonus = min(sentence_negative_score, 1.0)
            return explicit_score + (0.25 * weak_score) + context_bonus

        context_bonus = min(sentence_negative_score, 0.4)
        return weak_score + context_bonus

    def _passes_detection_threshold(self, evidence: ReasonEvidence) -> bool:
        if evidence.explicit_evidence_count > 0:
            return evidence.score >= self.min_score
        return evidence.score >= self.weak_min_score

    def _add_evidence(
        self,
        evidence: ReasonEvidence,
        sentence: str,
        matches: list[PatternMatch],
        score: float,
    ) -> None:
        evidence.score += score
        evidence.evidence_count += len(matches)
        evidence.explicit_evidence_count += sum(
            1 for match in matches if match.pattern.evidence_type == "explicit"
        )
        evidence.weak_evidence_count += sum(
            1 for match in matches if match.pattern.evidence_type == "weak"
        )

        if sentence not in evidence.supporting_sentences:
            if len(evidence.supporting_sentences) < self.max_supporting_sentences:
                evidence.supporting_sentences.append(sentence)

        for match in matches:
            if match.pattern.regex not in evidence.matched_patterns:
                evidence.matched_patterns.append(match.pattern.regex)

    @staticmethod
    def _uncertain_result(
        normalized_text: str,
        sentence_count: int,
        include_normalized_text: bool = False,
    ) -> DetectionResult:
        reasons = {
            OTHER_LABEL: {
                "label": OTHER_LABEL,
                "score": 0.0,
                "heuristic_confidence": 0.0,
                # Backward-compatible alias only; this is not a calibrated probability.
                "confidence": 0.0,
                "evidence_count": 0,
                "explicit_evidence_count": 0,
                "weak_evidence_count": 0,
                "supporting_sentences": [],
                "matched_patterns": [],
            }
        }
        return DetectionResult(
            labels=[OTHER_LABEL],
            reasons=reasons,
            has_multiple_reasons=False,
            sentence_count=sentence_count,
            normalized_text=normalized_text if include_normalized_text else None,
        )


def _print_example(review: str, result: DetectionResult) -> None:
    print("Review:")
    print(f"  {review}")
    print("Detected labels:")
    print(f"  {', '.join(result.labels)}")
    print("Evidence:")
    for label, evidence in result.reasons.items():
        print(
            "  "
            f"{label}: score={evidence['score']}, "
            f"heuristic_confidence={evidence['heuristic_confidence']}, "
            f"count={evidence['evidence_count']}"
        )
        for sentence in evidence["supporting_sentences"]:
            print(f"    - {sentence}")
    print()


if __name__ == "__main__":
    detector = NegativeReasonDetector()

    sample_reviews = [
        (
            "The acting was wooden and the dialogue felt painfully cheesy. "
            "Even the cast looked bored."
        ),
        (
            "I expected so much more from this director. The plot made no sense, "
            "dragged for two hours, and the ending was rushed and unsatisfying."
        ),
        (
            "The CGI looked cheap, the story was predictable, and every character "
            "felt flat and underdeveloped."
        ),
        "I did not enjoy this movie, but I cannot point to one clear reason.",
        "",
    ]

    for sample_review in sample_reviews:
        _print_example(sample_review, detector.analyze_review(sample_review))
