"""Cinematic layout presets and region resolution for spatial diffusion."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from mlx_spatial_diffusion.core.mask import Region


@dataclass
class LayoutSlot:
    """Represents a bounded geometric slot within a cinematic shot preset."""
    name: str
    box: List[float]  # [ymin, xmin, ymax, xmax]
    feather_radius: int = 4
    weight: float = 1.0
    aliases: List[str] = field(default_factory=list)


@dataclass
class ShotPreset:
    """A cinematic shot archetype defining spatial composition."""
    name: str
    description: str
    slots: Dict[str, LayoutSlot]
    default_environment: str = ""


# Built-in cinematic archetypes designed for 2-character dramatic interactions
CINEMATIC_PRESETS: Dict[str, ShotPreset] = {
    "two_shot_eye_level": ShotPreset(
        name="two_shot_eye_level",
        description="Standard balanced two-shot at eye level, subjects on left and right with headroom",
        slots={
            "left": LayoutSlot(
                name="left",
                box=[0.15, 0.05, 0.95, 0.48],
                feather_radius=4,
                aliases=["character_1", "left_character", "speaker_1"],
            ),
            "right": LayoutSlot(
                name="right",
                box=[0.15, 0.52, 0.95, 0.95],
                feather_radius=4,
                aliases=["character_2", "right_character", "speaker_2"],
            ),
        },
    ),
    "kneeling_ceremony": ShotPreset(
        name="kneeling_ceremony",
        description="Ceremony or knighting shot: kneeling subject on lower-left, elevated subject/throne on right",
        slots={
            "kneeling": LayoutSlot(
                name="kneeling",
                box=[0.30, 0.05, 0.98, 0.48],
                feather_radius=4,
                aliases=["supplicant", "knight", "soldier", "left", "character_1"],
            ),
            "standing": LayoutSlot(
                name="standing",
                box=[0.10, 0.48, 0.95, 0.95],
                feather_radius=4,
                aliases=["ruler", "king", "queen", "throne", "right", "character_2"],
            ),
        },
    ),
    "duel_confrontation": ShotPreset(
        name="duel_confrontation",
        description="Dynamic face-to-face confrontation or duel with wide dramatic stances",
        slots={
            "combatant_left": LayoutSlot(
                name="combatant_left",
                box=[0.18, 0.02, 0.95, 0.48],
                feather_radius=4,
                aliases=["left", "attacker", "character_1"],
            ),
            "combatant_right": LayoutSlot(
                name="combatant_right",
                box=[0.18, 0.52, 0.98, 0.98],
                feather_radius=4,
                aliases=["right", "defender", "character_2"],
            ),
        },
    ),
    "over_the_shoulder": ShotPreset(
        name="over_the_shoulder",
        description="Over-the-shoulder conversation shot: blurred back/shoulder in foreground left, focal subject in midground right",
        slots={
            "foreground": LayoutSlot(
                name="foreground",
                box=[0.25, 0.02, 0.98, 0.42],
                feather_radius=4,
                aliases=["shoulder", "back", "left", "character_1"],
            ),
            "focal": LayoutSlot(
                name="focal",
                box=[0.15, 0.42, 0.90, 0.95],
                feather_radius=4,
                aliases=["facing", "speaker", "right", "character_2"],
            ),
        },
    ),
    "hero_and_sidekick": ShotPreset(
        name="hero_and_sidekick",
        description="Hero dominating foreground/midground, companion positioned slightly behind on the right",
        slots={
            "hero": LayoutSlot(
                name="hero",
                box=[0.10, 0.05, 0.95, 0.60],
                feather_radius=4,
                aliases=["leader", "foreground", "left", "character_1"],
            ),
            "sidekick": LayoutSlot(
                name="sidekick",
                box=[0.25, 0.60, 0.90, 0.95],
                feather_radius=4,
                aliases=["companion", "background", "right", "character_2"],
            ),
        },
    ),
    "left_right_split": ShotPreset(
        name="left_right_split",
        description="Even vertical split dividing canvas into left and right halves",
        slots={
            "left": LayoutSlot(
                name="left",
                box=[0.05, 0.05, 0.95, 0.48],
                feather_radius=4,
                aliases=["side_a", "character_1"],
            ),
            "right": LayoutSlot(
                name="right",
                box=[0.05, 0.52, 0.95, 0.95],
                feather_radius=4,
                aliases=["side_b", "character_2"],
            ),
        },
    ),
}


class LayoutResolver:
    """Translates high-level scene directives and shot presets into concrete spatial Regions."""

    def __init__(self, custom_presets: Optional[Dict[str, ShotPreset]] = None):
        self.presets = dict(CINEMATIC_PRESETS)
        if custom_presets:
            self.presets.update(custom_presets)

    @classmethod
    def is_spatial_scene(cls, scene: Dict[str, Any]) -> bool:
        """Determines whether a scene requires spatial multi-region generation."""
        if "spatial_regions" in scene and scene["spatial_regions"]:
            return True
        layout = scene.get("layout", "").strip().lower()
        if layout and layout not in ("single", "none", "standard", "full"):
            return True
        if "characters_in_scene" in scene and isinstance(scene["characters_in_scene"], dict):
            if len(scene["characters_in_scene"]) >= 2:
                return True
        return False

    def resolve(
        self,
        scene: Dict[str, Any],
        characters: Optional[Dict[str, str]] = None,
        style_prefix: str = "",
        negative_prompt: str = "",
    ) -> Tuple[List[Region], str]:
        """Resolves a scene dictionary into a list of Region objects and a base environment prompt.

        Args:
            scene: Scene dictionary from script.json (contains 'layout', 'characters_in_scene', etc.)
            characters: Invariant character description mapping from script root
            style_prefix: Global aesthetic style prefix (e.g. 'cinematic anime, 8k')
            negative_prompt: Global negative prompt

        Returns:
            Tuple of (list_of_regions, base_prompt)
        """
        characters = characters or {}

        # 1. Base Environment Prompt
        base_prompt = (
            scene.get("environment")
            or scene.get("background")
            or scene.get("base_prompt")
            or ""
        )
        if not base_prompt and "visual_description" in scene:
            # If no explicit environment given, use visual_description as base context
            base_prompt = scene["visual_description"]

        if style_prefix and base_prompt:
            base_prompt = f"{style_prefix.strip('., ')}. {base_prompt.strip('., ')}"

        # 2. Case A: Explicit spatial_regions specified directly with bounding boxes
        if "spatial_regions" in scene and scene["spatial_regions"]:
            regions = []
            for item in scene["spatial_regions"]:
                prompt = self._build_character_prompt(item["prompt"], characters, style_prefix)
                region = Region(
                    box=item["box"],
                    prompt=prompt,
                    weight=item.get("weight", 1.0),
                    feather_radius=item.get("feather_radius", 4),
                    negative_prompt=item.get("negative_prompt", negative_prompt),
                )
                regions.append(region)
            return regions, base_prompt

        # 3. Case B: Preset-based layout resolution
        layout_name = scene.get("layout", "two_shot_eye_level").strip().lower()
        preset = self.presets.get(layout_name, self.presets.get("two_shot_eye_level"))

        char_inputs = scene.get("characters_in_scene", {})
        regions = self._match_slots_to_inputs(
            preset=preset,
            char_inputs=char_inputs,
            characters=characters,
            style_prefix=style_prefix,
            negative_prompt=negative_prompt,
        )

        return regions, base_prompt

    def _match_slots_to_inputs(
        self,
        preset: ShotPreset,
        char_inputs: Union[Dict[str, Any], List[Any]],
        characters: Dict[str, str],
        style_prefix: str,
        negative_prompt: str,
    ) -> List[Region]:
        """Matches user/LLM character inputs to predefined preset geometric slots."""
        regions = []
        preset_slots = list(preset.slots.values())

        if isinstance(char_inputs, list):
            # Positional matching: item 0 -> slot 0, item 1 -> slot 1
            for idx, item in enumerate(char_inputs):
                if idx >= len(preset_slots):
                    break
                slot = preset_slots[idx]
                prompt_str = item if isinstance(item, str) else item.get("prompt", "")
                prompt = self._build_character_prompt(prompt_str, characters, style_prefix)
                regions.append(
                    Region(
                        box=slot.box,
                        prompt=prompt,
                        weight=slot.weight,
                        feather_radius=slot.feather_radius,
                        negative_prompt=negative_prompt,
                    )
                )
            return regions

        if isinstance(char_inputs, dict):
            # Key/alias matching
            assigned_keys = set()
            for slot_name, slot in preset.slots.items():
                match_val = None
                # Check exact slot name
                if slot_name in char_inputs:
                    match_val = char_inputs[slot_name]
                    assigned_keys.add(slot_name)
                else:
                    # Check aliases
                    for alias in slot.aliases:
                        if alias in char_inputs:
                            match_val = char_inputs[alias]
                            assigned_keys.add(alias)
                            break

                if match_val is not None:
                    prompt_str = match_val if isinstance(match_val, str) else match_val.get("prompt", "")
                    prompt = self._build_character_prompt(prompt_str, characters, style_prefix)
                    regions.append(
                        Region(
                            box=slot.box,
                            prompt=prompt,
                            weight=slot.weight,
                            feather_radius=slot.feather_radius,
                            negative_prompt=negative_prompt,
                        )
                    )

            # If some inputs weren't matched by name/alias, fill remaining unassigned slots
            unassigned_slots = [s for s in preset_slots if s.box not in [r.box for r in regions]]
            remaining_keys = [k for k in char_inputs.keys() if k not in assigned_keys]

            for slot, k in zip(unassigned_slots, remaining_keys):
                match_val = char_inputs[k]
                prompt_str = match_val if isinstance(match_val, str) else match_val.get("prompt", "")
                prompt = self._build_character_prompt(prompt_str, characters, style_prefix)
                regions.append(
                    Region(
                        box=slot.box,
                        prompt=prompt,
                        weight=slot.weight,
                        feather_radius=slot.feather_radius,
                        negative_prompt=negative_prompt,
                    )
                )

        return regions

    def _build_character_prompt(
        self,
        raw_prompt: str,
        characters: Dict[str, str],
        style_prefix: str,
    ) -> str:
        """Enriches raw slot text with character anchors if referenced, plus styling."""
        resolved = raw_prompt.strip()

        # If raw_prompt is an exact key in characters mapping, use the full character description
        if resolved.lower() in {k.lower(): v for k, v in characters.items()}:
            for k, v in characters.items():
                if resolved.lower() == k.lower():
                    resolved = v
                    break
        else:
            # Check if any character name is mentioned, append their anchor features if helpful
            for char_name, char_desc in characters.items():
                if char_name.lower() in resolved.lower() and char_desc.lower() not in resolved.lower():
                    resolved = f"{resolved}. Character appearance: {char_desc}"

        parts = []
        if style_prefix:
            parts.append(style_prefix.strip("., "))
        if resolved:
            parts.append(resolved.strip("., "))

        return ". ".join(parts)
