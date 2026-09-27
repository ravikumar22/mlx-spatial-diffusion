"""Unit tests for LayoutResolver and cinematic shot presets."""

import unittest
from mlx_spatial_diffusion.layout.resolver import LayoutResolver, CINEMATIC_PRESETS


class TestLayoutResolver(unittest.TestCase):
    def test_presets_exist(self):
        self.assertIn("two_shot_eye_level", CINEMATIC_PRESETS)
        self.assertIn("kneeling_ceremony", CINEMATIC_PRESETS)
        self.assertIn("duel_confrontation", CINEMATIC_PRESETS)
        self.assertIn("over_the_shoulder", CINEMATIC_PRESETS)
        self.assertIn("hero_and_sidekick", CINEMATIC_PRESETS)
        self.assertIn("left_right_split", CINEMATIC_PRESETS)

    def test_is_spatial_scene(self):
        resolver = LayoutResolver()
        
        # Negative cases
        self.assertFalse(resolver.is_spatial_scene({"visual_description": "A quiet mountain landscape"}))
        self.assertFalse(resolver.is_spatial_scene({"layout": "single", "visual_description": "Single warrior"}))
        self.assertFalse(resolver.is_spatial_scene({"layout": "none"}))
        
        # Positive cases
        self.assertTrue(resolver.is_spatial_scene({"layout": "two_shot_eye_level"}))
        self.assertTrue(resolver.is_spatial_scene({"spatial_regions": [{"box": [0, 0, 1, 1], "prompt": "test"}]}))
        self.assertTrue(resolver.is_spatial_scene({
            "characters_in_scene": {
                "left": "Knight",
                "right": "King"
            }
        }))

    def test_resolve_two_shot_eye_level(self):
        resolver = LayoutResolver()
        scene = {
            "layout": "two_shot_eye_level",
            "environment": "Grand throne room with vaulted ceiling",
            "characters_in_scene": {
                "left": "Young soldier in steel armor",
                "right": "Old king with gold crown",
            }
        }
        regions, base_prompt = resolver.resolve(scene, style_prefix="cinematic anime")
        
        self.assertEqual(base_prompt, "cinematic anime. Grand throne room with vaulted ceiling")
        self.assertEqual(len(regions), 2)
        
        # Check left slot
        r_left = regions[0]
        self.assertEqual(r_left.box, [0.15, 0.05, 0.95, 0.48])
        self.assertIn("soldier in steel armor", r_left.prompt)
        self.assertIn("cinematic anime", r_left.prompt)
        
        # Check right slot
        r_right = regions[1]
        self.assertEqual(r_right.box, [0.15, 0.52, 0.95, 0.95])
        self.assertIn("king with gold crown", r_right.prompt)

    def test_resolve_kneeling_ceremony_aliases(self):
        resolver = LayoutResolver()
        scene = {
            "layout": "kneeling_ceremony",
            "background": "Cathedral stained glass windows",
            "characters_in_scene": {
                "knight": "Kneeling knight bowing head",
                "king": "Seated emperor holding scepter",
            }
        }
        regions, base_prompt = resolver.resolve(scene)
        self.assertEqual(len(regions), 2)
        self.assertIn("stained glass", base_prompt)
        
        # "knight" should match kneeling slot
        knight_reg = [r for r in regions if "Kneeling knight" in r.prompt][0]
        self.assertEqual(knight_reg.box, [0.30, 0.05, 0.98, 0.48])
        
        # "king" should match standing slot
        king_reg = [r for r in regions if "Seated emperor" in r.prompt][0]
        self.assertEqual(king_reg.box, [0.10, 0.48, 0.95, 0.95])

    def test_resolve_character_anchor_mapping(self):
        resolver = LayoutResolver()
        characters = {
            "arjuna": "legendary archer, sapphire blue dhoti, Gandiva bow in hand",
            "krishna": "divine charioteer, yellow pitambara silk, peacock feather in hair",
        }
        scene = {
            "layout": "two_shot_eye_level",
            "characters_in_scene": {
                "left": "arjuna",
                "right": "Lord Krishna smiling serenely",
            }
        }
        regions, base_prompt = resolver.resolve(scene, characters=characters)
        self.assertEqual(len(regions), 2)
        
        # Exact key match for left
        self.assertIn("Gandiva bow in hand", regions[0].prompt)
        # Mention match for right
        self.assertIn("Lord Krishna smiling serenely", regions[1].prompt)
        self.assertIn("peacock feather in hair", regions[1].prompt)

    def test_resolve_explicit_spatial_regions(self):
        resolver = LayoutResolver()
        scene = {
            "environment": "Misty forest glade",
            "spatial_regions": [
                {
                    "box": [0.2, 0.1, 0.9, 0.5],
                    "prompt": "Wizard casting a spell",
                    "feather_radius": 5,
                },
                {
                    "box": [0.2, 0.5, 0.9, 0.9],
                    "prompt": "Dragon breathing smoke",
                    "feather_radius": 3,
                }
            ]
        }
        regions, base_prompt = resolver.resolve(scene)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].box, [0.2, 0.1, 0.9, 0.5])
        self.assertEqual(regions[0].feather_radius, 5)
        self.assertEqual(regions[1].box, [0.2, 0.5, 0.9, 0.9])
        self.assertEqual(regions[1].feather_radius, 3)


if __name__ == "__main__":
    unittest.main()
